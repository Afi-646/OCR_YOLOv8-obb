#  Colab Notebooks

1. [溫度計辨識](https://colab.research.google.com/drive/1cmM4QckOGzwUqcLHn1-dgiRu_pFuGjqN?usp=sharing "溫度計辨識搭配互動式表單")
2. [Roboflow_Ocrscale辨識](https://colab.research.google.com/drive/1Pt54Fsql51jW3LhHacS7tHOvPj_D0A4H?usp=sharing "Roboflow+格式化")
3. [自訓YOLOv8-OBB模型偵測](https://colab.research.google.com/drive/1gsB5WnxcvgGL1kxHi_BJ1sxXVwm9D4SC?usp=sharing "OBB模型，偵測螢幕，數字0~9")
4. [自訓YOLOv8模型偵測+搭配Roboflow資料集](https://colab.research.google.com/drive/1E1old6SiFOBw85dFM9f3jlxy5RWQqtQP?usp=sharing "偵測0~9，小數點-，kwh")

##  溫度計辨識
使用Hugging Face API串接
<img width="901" height="429" alt="image" src="https://github.com/user-attachments/assets/e6e6104c-0d6f-4099-961e-dbc99ef8baf4" />

使用方法：
1.  上傳圖片到Notebook檔案
2.  在互動式表單輸入提示詞與圖片位置
3.  執行並檢視輸出結果

##  Roboflow Ocrscale辨識
使用Roboflow API串接，適用經過梯形校正或旋轉，讀數由左至右呈現的圖片
<img width="896" height="151" alt="image" src="https://github.com/user-attachments/assets/c8377827-ff84-4deb-a52f-ed163ca47789" />

使用方法：
1.  上傳圖片到Notebook檔案
2.  在互動式表單輸入檔案名稱
3.  執行並檢視輸出結果

## 自訓YOLOv8-OBB模型偵測
以原圖標註。訓練模型判讀螢幕、數值、符號，再透過後處理腳本讀取最合理的溫度
<img width="984" height="332" alt="image" src="https://github.com/user-attachments/assets/159971bd-5b50-42e0-b014-48f119833496" />
<img width="703" height="446" alt="image" src="https://github.com/user-attachments/assets/0a8980b8-c7c7-4f0a-8c72-255427d42ffe" />
![train_batch3781](https://github.com/user-attachments/assets/f1273d63-4506-43e6-9146-729093392bbd)

使用方法：
1.  下載YOLOv8-obb資料夾內的best.pt檔案，上傳至Notebook
2.  上傳圖片到Notebook檔案
3.  在互動式表單輸入檔案名稱，設定用於篩選結果的信心值
4.  執行並檢視輸出結果
---
**新、舊版差異**

舊版只顯示一個信心值最高的數值，新版針對信心值相近的數值給出兩種以上的組合，並回傳使用者選擇flag

##  自訓YOLOv8模型偵測+搭配Roboflow資料集
使用Ocrscale辨識同樣的資料集，移除干擾資料後重新訓練(沒有混合自有資料)
<img width="536" height="359" alt="image" src="https://github.com/user-attachments/assets/4cf10523-abb6-4143-b8e2-818303b83f32" />
![val_batch1_labels](https://github.com/user-attachments/assets/f21d13d7-ddf0-48ff-805b-7d241ecbb0d0)

使用方法：
1.  下載YOLOv8資料夾內的best.pt檔案，上傳至Notebook
2.  依序執行，根據指示透過按鈕上傳圖片
3.  檢視輸出結果
4.  額外驗證——透過YOLO進行推論，產出Labels框選圖與定位.txt檔 




