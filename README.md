# OCR_YOLOv8-obb
目前使用YOLOv8-Oriented bounding boxes偵測溫度計數值

### 需求
* 讀取圖片中溫度計的數值，只擷取數字部分（不含攝氏符號，如：91.2）
* 異常：低於設定的區間、讀不到數值，傳送flag到web端，呼叫線上推理服務api


### 使用過的方案
* [溫度計圖片辨識器](https://cjian2025-rc-temature1025.hf.space "溫度計圖片辨識器")
  
  使用promt：請擷取溫度計顯示的溫度。請僅輸出一個數字（例如 91.9），不得包含任何說明文字或符號。

  使用ROI圖片(針對螢幕範圍進行梯形校正)
  
  正確率：84% (213/253)

* docTR (使用ROI過的圖片)

  辨識率：73.12% (185/253)

  正確率：59.68% (151/253)

* [YOLOv8](https://colab.research.google.com/drive/1E1old6SiFOBw85dFM9f3jlxy5RWQqtQP?usp=sharing "TestM-1.colab")

  模型位置：YOLOv8/best.pt
  
  推估正確率：70~83%

  
* [YOLOv8-obb](https://colab.research.google.com/drive/1gsB5WnxcvgGL1kxHi_BJ1sxXVwm9D4SC?usp=sharing "TestM-obb.colab")

  模型位置：YOLOv8-obb/best.pt

  推估正確率： 約70%~80%
