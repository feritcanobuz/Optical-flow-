import numpy as np
import cv2
import time

# Lucas-Kanade optik akış parametreleri
lk_params = dict(winSize=(15, 15),  # Pencere boyutu
                 maxLevel=2,  # Piramit seviyesi
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Durdurma kriteri

# Yeni özelliklerin tespiti için parametreler
feature_params = dict(maxCorners=100,  # Daha fazla köşe tespiti
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 20  # İz uzunluğu
detect_interval = 1  # Tespit aralığı
trajectories = []  # Yörüngeler listesi
frame_idx = 0  # Kare indeksi

cap = cv2.VideoCapture(0)  # Kamerayı başlat

# Kamera doğru şekilde açıldı mı kontrol et
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# İlk çerçeve
suc, frame = cap.read()
if not suc:  # Eğer kare alınamadıysa çık
    print("İlk kare alınamadı!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # İlk kareyi griye dönüştür

while True:
    start = time.time()  # FPS hesaplamak için başlangıç zamanı
    suc, frame = cap.read()  # Yeni kareyi oku
    if not suc:  # Eğer kare alınamadıysa döngüyü bitir
        print("Kare alınamadı!")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Renkli kareyi griye dönüştür
    img = frame.copy()

    # Optik akış hesapla
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)  # Ters optik akış
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1  # Geçerli noktaları al

        new_trajectories = []

        # Yörüngeleri güncelle
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue  # Geçersiz noktaları geç
            trajectory.append((x, y))

            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)

            # Yeni tespit edilen nokta
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        trajectories = new_trajectories

        # Yörüngeleri çiz
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0), 2)
        cv2.putText(img, 'Track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Yeni özellikler tespit etme
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255  # Bütün pikselleri beyaz yap

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)  # Yeni noktalar için siyah daireler çiz

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # FPS hesapla
    end = time.time()
    fps = 1 / (end - start)

    # Sonuçları göster
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Eğer görüntü alınabiliyorsa, ekrana göster
    if frame is not None:
        cv2.imshow('Optical Flow', img)
        cv2.imshow('Mask', mask)

    # 'q' tuşuna basarak çıkabilirsin
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
