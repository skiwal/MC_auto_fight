from pathlib import Path
from ultralytics import YOLO


def main():
    # =========================
    # 1. 基本配置
    # =========================

    data_yaml = "D:/code/MC_auto_fight/yolo_dataset/data.yaml"

    # 推荐先用 yolo26n.pt，速度快，适合 Minecraft 实时检测
    # 如果效果不够，再换成 yolo26s.pt
    model_name = "yolo26n.pt"

    project_dir = "runs/mc_yolo26"
    run_name = "mc_mob_yolo26n"

    # =========================
    # 2. 加载 YOLO26 模型
    # =========================

    model = YOLO(model_name)

    # =========================
    # 3. 开始训练
    # =========================

    train_results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,          # 有 NVIDIA GPU 用 0；没有 GPU 改成 "cpu"
        workers=4,
        project=project_dir,
        name=run_name,
        patience=30,
        cache=True,
        amp=True,
        seed=42,
        optimizer="auto",
        close_mosaic=10,
        val=True
    )

    # =========================
    # 4. 找到 best.pt
    # =========================

    save_dir = Path(train_results.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"
    last_model_path = save_dir / "weights" / "last.pt"

    print("\nTraining finished.")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")

    # =========================
    # 5. 使用 best.pt 做验证
    # =========================

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=data_yaml,
        imgsz=640,
        device=0
    )

    print("\nValidation metrics:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50:    {metrics.box.map50}")
    print(f"mAP75:    {metrics.box.map75}")


if __name__ == "__main__":
    main()