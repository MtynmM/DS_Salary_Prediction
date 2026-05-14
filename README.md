
🚀 هوش مصنوعی پیش‌بینی حقوق متخصصان داده (DS Salary Predictor)
از نوت‌بوک‌ ها تا نرم‌افزار مقیاس‌ پذیر و ماژولار

📌 مروری بر پروژه
این پروژه یک سامانه End-to-End و ماژولار برای مهندسی داده، آموزش مدل‌های یادگیری ماشین و استقرار وب‌ سرویس است که با هدف پیش‌ بینی حقوق متخصصان حوزه داده (Data Scientists، ML Engineers و غیره) طراحی شده است. تمرکز اصلی این معماری، گذر از اسکریپت‌های یکپارچه (Monolithic Notebooks) به سمت یک نرم‌افزار مقیاس‌ پذیر (Scalable Software) با رعایت اصول مهندسی نرم‌افزار می‌ باشد.

🏗️ معماری سیستم (System Architecture)
برای حفظ سادگی در توسعه و سهولت در استقرار، معماری Modular Monolith انتخاب شده است. لایه‌ها (Concerns) کاملاً از یکدیگر ایزوله شده‌اند تا برای مثال تغییر در دیتابیس یا مدل، نیازی به بازنویسی لایه API نداشته باشد.

text
DS_Salary_Prediction
┣ 📁 data/                     # Persistence Layer (SQLite DB & Joblib Artifacts)
┣ 📁 src/
┃   ┣ 📁 data_pipeline/        # Data Engineering & ETL (SQLAlchemy)
┃   ┣ 📁 models/               # ML Engine, Training & Evaluation
┃   ┗ 📁 api/                  # Serving Layer (FastAPI, Pydantic)
┗ 📄 main.py                   # Central Orchestrator
🔑 تصمیمات کلیدی معماری (Architecture Decisions)
چالش	رویکرد صنعتی	مزیت
تزریق وابستگی (Dependency Injection)	تزریق شیء Predictor در زمان اجرا به جای هاردکد کردن مسیر مدل در API	افزایش قابلیت تست‌پذیری (Testability) برای نوشتن Mock Tests
مدیریت ورودی‌های پیش‌ بینی‌ نشده (OOV)	ساخت دیکشنری mappings.joblib در لایه ETL برای نگاشت هوشمند ورودی‌های ناشناس به ویژگی "Other"	جلوگیری از خطاهای 500 Internal Server Error و افزایش پایداری در محیط Production
اعتبارسنجی لبه سیستم (Edge Validation)	استفاده از Schemaهای سخت‌گیرانه Pydantic در لایه روتر FastAPI	اعتبارسنجی نوع و محدوده داده‌ها قبل از ورود به چرخه پیش‌ بینی و تضمین امنیت سیستم
غلبه بر (Curse of Dimensionality)	انتخاب رگرسیون خطی به عنوان Champion Model به جای Random Forest	جلوگیری از overfitting در ماتریس‌های خلوت (Sparse Matrices) و اثبات اصل Occam's Razor
| الگوریتم          | میانگین R² | RMSE (دلار) | وضعیت                  |
|--------------------|------------|-------------|----------------------  |
| Linear Regression  | 0.5253     | $52,532     | ✅ Champion (Deployed) |
| Random Forest      | 0.4524     | $57,302     | ❌ Rejected (Overfit)  |
مدل رگرسیون خطی با اختصاص وزن مستقیم، عملکرد بهتری نسبت به مدل پیچیده‌تر Random Forest داشته و به عنوان مدل نهایی انتخاب شده است.

🛠️ پشته تکنولوژی (Tech Stack)
لایه	فناوری‌ها
Serving & API	FastAPI, Uvicorn, Pydantic
Machine Learning	Scikit-Learn, NumPy, Joblib (Optimized Serialization)
Data Persistence	Pandas, SQLAlchemy, SQLite
Design Patterns	Dependency Injection, Orchestrator, Factory Pattern

⚡ راهنمای نصب و اجرا (Quick Start)
bash
# ۱. کلون مخزن
git clone https://github.com/MtynmM/DS_Salary_Prediction.git
cd DS_Salary_Prediction

# ۲. نصب نیازمندی‌ها
pip install -r requirements.txt

# ۳. اجرای ارکستراتور (ETL + Training + Serving)
python main.py
با اجرای دستور فوق، سیستم به طور خودکار پایپ‌لاین را در صورت عدم وجود Artifactها اجرا کرده و سرور را در localhost:8000 بالا می‌آورد.

🔗 مستندات تعاملی API (Swagger)
پس از اجرا، مستندات کامل API در آدرس زیر در دسترس است:
👉 http://127.0.0.1:8000/docs

🗺️ نقشه راه توسعه (Roadmap to MLOps Level 1)
برای ارتقای این MVP به یک پلتفرم کاملاً بالغ در سطح MLOps، موارد زیر در دستور کار قرار دارند:

کانتینرسازی محیط با Docker و docker-compose

پیاده‌سازی تست‌های خودکار (Unit/Integration Tests) با Pytest

اضافه کردن MLflow برای Model Registry و Experiment Tracking

راه‌اندازی CI/CD Pipeline با GitHub Actions


👨‍💻 توسعه‌دهنده
متین محمدی (Matin Mohammadi) - مهندس نرم‌افزار

[![GitHub](https://img.shields.io/badge/GitHub-MtynmM-181717?style=flat-square&logo=github)](https://github.com/MtynmM)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat-square&logo=fastapi&logoColor=white)]()


⭐ حمایت شما
اگر این پروژه برایتان مفید بود، لطفاً با ستاره دادن به مخزن، از آن حمایت کنید. 🙏