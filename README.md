### 整體專案需求
因為APP端是早期的cordova開發，目前已有想法要更換新的開發語言，所以現階段過渡期規劃希望另外單純開發一個可以被呼叫 **OCR辨識的元件** 以及  **WEB端OCR API** 功能，未來新開發後還可以繼續使用，

使用場域說明如下：

1.  使用者使用手機(iOS、Android版)對烹煮菜色、溫度計拍照並辨識溫度計數值(因為照片是留存紀錄用，所以溫度計可能只是整體畫面的一小部分、如圖)。
2.  拍照場域可能無網路，結果檔案、資料會暫時留存在手機上，直到有網路地點後就會自動上傳資料，所以**APP端辨識只能在本機作用**。
3.  新元件拍照時可以根據不同溫度計類型框住溫度數值並即時顯示辨識數字後，按下拍照回傳
   
    1.  原始照片、
    2.  辨識結果數字、
    3.  框住的數值照片、
    4.  辨識度參考值(1~10數字越高越正確)。

5.  定時排程程式自動將辨識度差的資料，PK+框住數值照片(BASE64)傳送至OCR API並轉送至外部雲端AI辨識平台或是自行辨識訓練平台再次辨識後回傳PK+辨識後數值。

希望能協助開發有兩部分，如圖紅色虛線框住：
1.  APP端OCR辨識元件。
2.  WEB端OCR API。


<img width="1149" height="769" alt="d50914e1-8c4d-4db1-9e90-a8fae72af0c5" src="https://github.com/user-attachments/assets/0b88d6f3-d577-4540-9bda-8e44c59f4832" />


### 影像辨識需求
* 讀取圖片中溫度計的數值，只擷取數字部分（不含攝氏符號，如：91.2）
* 異常：低於設定的區間、讀不到數值，傳送flag到web端，呼叫線上推理服務api

### 測試過的方案
* [溫度計圖片辨識器](https://cjian2025-rc-temature1025.hf.space "溫度計圖片辨識器")
  
  使用promt：請擷取溫度計顯示的溫度。請僅輸出一個數字（例如 91.9），不得包含任何說明文字或符號。

  使用ROI圖片(針對螢幕範圍進行梯形校正，沒有其他處理)
  
  正確率：84% (213/253)

  >   使用腳本測試階段曾嘗試使用多管道(多個ROI灰階策略)投票，但灰階結果不理想

* docTR (使用ROI過的圖片)

  辨識率：73.12% (185/253)

  正確率：59.68% (151/253)

* [YOLOv8](https://colab.research.google.com/drive/1E1old6SiFOBw85dFM9f3jlxy5RWQqtQP?usp=sharing "TestM-1.colab")

  模型位置：YOLOv8/best.pt
  
  推估正確率：70~83%

  
* [YOLOv8-obb](https://colab.research.google.com/drive/1gsB5WnxcvgGL1kxHi_BJ1sxXVwm9D4SC?usp=sharing "TestM-obb.colab")

  模型位置：YOLOv8-obb/best.pt

  推估正確率： 約70%~80%

*  [線上偵測Roboflow](https://universe.roboflow.com/hstech/ocrscale "ocrscale")
  
   要註冊、有額度限制
