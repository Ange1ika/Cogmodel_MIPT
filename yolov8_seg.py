import cv2
import torch
import intel_extension_for_pytorch as ipex

from ultralytics import YOLO


print(torch.xpu.is_available())


# Загрузка модели YOLOv8-сегментация
model = YOLO('yolov8n-seg.pt')  # Вы можете использовать любые модели YOLOv8 (n, m, l, x)

# Открытие видеофайла
video_path = '/home/angelika/Desktop/cogmodel/movie.mkv'
cap = cv2.VideoCapture(video_path)

# Определение выходного видео (кодек и параметры)
output_path = '/home/angelika/Desktop/cogmodel/movie_res.mkv'
fourcc = cv2.VideoWriter_fourcc(*'x264')  # Используется кодек 'x264'
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Получаем FPS исходного видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Проверка наличия GPU
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(device)

# Перенос модели на устройство
model.to(device)

# Чтение видео покадрово и инференс
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Конец видео
    
    # Выполнение инференса на текущем кадре
    results = model(frame)
    
    # Извлечение результата (можно также обрабатывать результат по необходимости)
    result_frame = results[0].plot()  # Отрисовка масок и сегментаций на кадре
    
    # Запись обработанного кадра в выходное видео
    out.write(result_frame)

    #plt.imshow(result_frame)
    #plt.show()

    # Опционально: Выводить кадр в реальном времени
    cv2.imshow('YOLOv8-Seg Inference', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Остановить, если нажата клавиша 'q'

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()


#import matplotlib.pyplot as plt


