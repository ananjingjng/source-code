from ultralytics import YOLO

if __name__ == '__main__':

    ############## 这是train的代码 ##############
    # model = YOLO(r"ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v10/yolov10n.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v5/yolov5.yaml")
    model = YOLO(r"ultralytics/cfg/models/v6/yolov6.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v3/yolov3.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v9/yolov9t.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-early.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF-CFE.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-RCSOSA.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-RCSOSA-SDI.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-early-RCSOSA-Detect_DBB.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-early-DWR.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-early-EMA-DWR.yaml")
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-SDI.yaml")
    model.train(data=r"ultralytics/cfg/datasets/FLIR.yaml", batch=64, epochs=150, project='runs/earlyFLIR/train=2', name='RGBv6', amp=False, workers=8, 
                   # optimizer='SGD',  # Optimizer
                   # cos_lr=True,  # Cosine LR Scheduler
                   # lr0=0.002
                  # 训练
               ) 
    ########### 这是val和predict的代码 ##############
    # model = YOLO(r"./best.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # 验证
    # model.predict(source=r"Datasets/llvip/images/test", save=True)  #   检测
