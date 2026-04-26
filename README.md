# ROS2 Autonomous Traffic Robot

> A ROS2-based robotic perception system for real-time traffic sign and traffic light recognition in Gazebo simulation.

> 基於 ROS2 的機器人視覺感知系統，可在 Gazebo 模擬環境中即時辨識交通標誌與紅綠燈。

---

## Project Overview

This project implements an autonomous traffic perception pipeline for TurtleBot3 in ROS2/Gazebo simulation.  
The robot uses deep learning-based computer vision models to detect traffic signs and classify traffic light states, then executes corresponding behavior commands through a decision-making module.

本專案實作一套用於 TurtleBot3 的自主交通感知系統，部署於 ROS2/Gazebo 模擬環境中。  
機器人透過深度學習視覺模型辨識交通標誌與紅綠燈狀態，並經由決策模組執行對應控制行為。

---

## Key Features

- Real-time traffic sign detection using YOLOv8  
  使用 YOLOv8 進行即時交通標誌偵測 
- Traffic light state classification using EfficientNet
  使用 EfficientNet 進行紅綠燈分類  
- Behavior decision logic for autonomous robot response
  建立自動決策邏輯控制機器人行為  
- Integrated ROS2 node communication pipeline
  完整 ROS2 節點通訊流程整合 
- Full Gazebo simulation testing environment
  可於 Gazebo 模擬環境完整測試
    

---

## System Architecture

### Perception and Control Pipeline

1. Subscribe to TurtleBot3 camera image stream
   訂閱 TurtleBot3 相機影像串流  
2. Perform traffic sign detection using YOLOv8
   使用 YOLOv8 偵測交通標誌  
3. Perform traffic light classification using EfficientNet
   使用 EfficientNet 分類紅綠燈狀態  
4. Fuse perception results into decision-making logic
   將感知結果整合進決策邏輯  
5. Publish velocity commands to `/cmd_vel`
   發布速度控制至 `/cmd_vel`  

---

## Technical Stack

| Category | Technology |
|---------|------------|
| Robot Framework | ROS2 Humble |
| Simulator | Gazebo |
| Programming Language | Python |
| Object Detection | YOLOv8 |
| Image Classification | EfficientNet-B0 |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |

---

## Supported Traffic Perception Classes

### Traffic Signs
- Stop Sign  
- Speed Limit  
- Crosswalk  
- Left Turn  
- Right Turn  

### Traffic Lights
- Red  
- Yellow  
- Green  

---

## Behavior Decision Logic

| Perception Result | Robot Action |
|------------------|-------------|
| Red Light | Stop |
| Yellow Light | Slow Down |
| Green Light | Move Forward |
| Stop Sign | Temporary Stop |
| Left Turn Sign | Turn Left |
| Right Turn Sign | Turn Right |
| Speed Limit | Adjust Speed |

---

## Engineering Challenges

### Robust Traffic Light Recognition
Traditional OpenCV color thresholding showed poor robustness under varying lighting conditions.  
To improve classification accuracy, the pipeline was redesigned using EfficientNet-based image classification.

傳統 OpenCV 色彩閾值方法在不同光照條件下穩定性不足。  
因此改採 EfficientNet 深度學習分類器提升辨識準確率。

---

### Similar Traffic Sign Appearance
Left-turn and right-turn signs exhibited high visual similarity, causing detection confusion.  
Dataset refinement and training optimization were applied to improve model discrimination.

左轉與右轉標誌外觀高度相似，容易造成辨識混淆。  
透過資料集優化與訓練調整改善模型區辨能力。

---

### Decision Stability
False positives occasionally caused unstable robot behavior.  
Confidence thresholding and temporal filtering were introduced to stabilize decision outputs.

誤偵測會導致機器人行為不穩定。  
因此加入 confidence threshold 與 temporal filtering 穩定決策結果。

---

## Demonstration Results

- Successfully performs real-time traffic perception in Gazebo simulation  
- Robot executes correct driving behavior according to detected traffic conditions  
- Achieves stable perception-to-control closed-loop integration  

- 成功於 Gazebo 中即時完成交通感知  
- 機器人可依辨識結果執行正確控制行為  
- 完成穩定的 perception-to-control 閉環系統整合  

---

## Future Improvements

- Integrate SLAM for environment mapping
- 整合 SLAM 建立環境地圖 
- Add Nav2 for autonomous path planning
- 加入 Nav2 完成自主路徑規劃
- Deploy to real TurtleBot3 hardware
- 部署至實體 TurtleBot3 
- Explore sim-to-real transfer optimization
- 研究 Sim-to-Real Transfer 優化 

---

## Demo Video

