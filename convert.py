from ultralytics import YOLO

model = YOLO("frc.pt")

model.export(format="onnx", opset=12)
#results = model(source=0, show=True, conf=0.1, save=True)