import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

# =========================
# READ DATA SET
# =========================
df = pd.read_csv(r"consumer_electronics_sales_data.csv")

# =========================
# DATA CLEAN
# =========================
le_category = LabelEncoder()
le_brand = LabelEncoder()

df['ProductCategory'] = le_category.fit_transform(df['ProductCategory'])
df['ProductBrand'] = le_brand.fit_transform(df['ProductBrand'])
df.drop('ProductID', axis=1, inplace=True)

# =========================
# FIT DATA
# =========================
X = df.drop('PurchaseIntent', axis=1)
y = df['PurchaseIntent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rn_cls = RandomForestClassifier(max_depth=10, random_state=0)
rn_cls.fit(X_train, y_train)

@app.get("/", response_class=HTMLResponse)
def hi():
    return """
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PurchaseIntent</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(to right, #f9f9f9, #e0f7fa);
                margin: 0;
                padding: 0;
                animation: fadeIn 1s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            header {
                background-color: #4CAF50;
                color: white;
                padding: 10px 0;
                text-align: center;
                font-size: 28px;
                animation: slideIn 0.5s forwards;
            }
            @keyframes slideIn {
                from { transform: translateY(-20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .container {
                width: 80%;
                margin: 20px auto;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            input[type="text"], input[type="number"] {
                width: calc(100% - 20px);
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 5px;
                transition: border 0.3s;
            }
            input[type="text"]:focus, input[type="number"]:focus {
                border: 1px solid #4CAF50;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
            #response {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            footer {
                text-align: center;
                margin-top: 20px;
                font-size: 16px;
                color: #777;
                animation: fadeIn 1s ease-in-out 0.5s forwards;
            }
        </style>
    </head>
    <body>
        <header>
            Purchase Intent
        </header>
        <div class="container">
            <form id="purchaseIntentForm">
                <label for="productCategory">فئة المنتج:</label>
                <input type="text" id="productCategory" name="PrCa" value="Smart Watches" required>

                <label for="productBrand">علامة المنتج:</label>
                <input type="text" id="productBrand" name="PrBr" value="Samsung" required>

                <label for="productPrice">سعر المنتج:</label>
                <input type="number" id="productPrice" name="PrPr" value="980.39" required step="0.01">

                <label for="customerAge">عمر العميل:</label>
                <input type="number" id="customerAge" name="CuAge" value="35" required>

                <label for="customerGender">جنس العميل (1 للذكور، 0 للإناث):</label>
                <input type="number" id="customerGender" name="CuGe" value="1" required>

                <label for="purchaseFrequency">تكرار الشراء:</label>
                <input type="number" id="purchaseFrequency" name="PuFr" value="7" required>

                <label for="customerSatisfaction">رضا العميل:</label>
                <input type="number" id="customerSatisfaction" name="CuSa" value="2" required>

                <button type="submit">إرسال</button>
            </form>
            <div id="response"></div>
        </div>
        <footer>
            SUHIB ALFURJANI
        </footer>

        <script>
            document.getElementById('purchaseIntentForm').addEventListener('submit', async function(event) {
                event.preventDefault(); // منع إعادة تحميل الصفحة
                const formData = new FormData(this);
                const params = new URLSearchParams(formData).toString();
                const apiUrl = `http://127.0.0.1:8000/PurchaseIntent?${params}`;

                try {
                    const response = await fetch(apiUrl);
                    const data = await response.json();
                    document.getElementById('response').innerText = `الاستجابة: ${data.message}`;
                } catch (error) {
                    console.error('حدث خطأ:', error);
                    document.getElementById('response').innerText = 'حدث خطأ أثناء الاتصال بالـ API. تحقق من البيانات المدخلة وتأكد من أن الـ API قيد التشغيل.';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/PurchaseIntent")
def predict_purchase(PrCa: str, PrBr: str, PrPr: float, CuAge: int, CuGe: int, PuFr: int, CuSa: int):
    # تحويل المدخلات باستخدام LabelEncoder
    try:
        input_data = [
            le_category.transform([PrCa])[0],
            le_brand.transform([PrBr])[0],
            PrPr,
            CuAge,
            CuGe,
            PuFr,
            CuSa
        ]
    except ValueError as e:
        return {"message": f"خطأ: {str(e)}"}

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = rn_cls.predict(input_data_reshaped)
    if prediction[0] == 0:
        return {"message": "No Purchase"}
    else:
        return {"message": "Purchase"}
