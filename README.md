🚀 AI Salary Predictor: Modular MVP Pipeline

این پروژه یک سیستم End-to-End و ماژولار برای مهندسی داده، آموزش مدل‌های یادگیری ماشین و استقرار وب‌سرویس جهت پیش‌بینی حقوق متخصصان داده (Data Scientists, ML Engineers و ...) است. تمرکز اصلی این معماری، گذر از اسکریپت‌های یکپارچه (Monolithic Notebooks) به سمت یک نرم‌افزار مقیاس‌پذیر (Scalable Software) با رعایت اصول مهندسی نرم‌افزار می‌باشد.

🏛️ معماری سیستم (System Architecture)

برای حفظ سادگی در توسعه و سهولت در استقرار، معماری Modular Monolith انتخاب شده است. لایه‌ها (Concerns) کاملاً از یکدیگر ایزوله شده‌اند (Decoupled) تا تغییر در دیتابیس یا مدل، نیازی به بازنویسی لایه API نداشته باشد.

```text
📦 DS_Salary_Prediction
┣ 📂 data/               # Persistence Layer (SQLite DB & Joblib Artifacts)
┣ 📂 src/
┃ ┣ 📂 data_pipeline/    # Data Engineering & ETL (SQLAlchemy)
┃ ┣ 📂 models/           # ML Engine, Training & Evaluation
┃ ┗ 📂 api/              # Serving Layer (FastAPI, Pydantic)
┗ 📜 main.py             # Central Orchestrator
```

🧠 تصمیمات کلیدی معماری (Architecture Decisions)

به عنوان مهندس نرم‌افزار، چالش‌های زیر با رویکردهای استاندارد صنعت برطرف شده‌اند:

۱. اصل تزریق وابستگی (Dependency Injection) در API

به جای هاردکد کردن (Hard-coding) مسیر مدل در API، شیء Predictor در زمان اجرا (Runtime) به تابع سازنده create_app تزریق می‌شود. این کار قابلیت تست‌پذیری (Testability) سیستم را در آینده برای نوشتن Mock Testها به شدت افزایش می‌دهد.

۲. پایداری در محیط عملیاتی با مدیریت OOV

یکی از دلایل اصلی شکست مدل‌های ML در محیط Production، دریافت داده‌های پیش‌بینی‌نشده (Out-of-Vocabulary) است. در لایه ETL، دیکشنریِ mappings.joblib ساخته می‌شود تا ورودی‌های ناشناس کاربر در محیط Inference با ظرافت (Gracefully) به ویژگی "Other" نگاشت شوند و از کرش کردن (500 Internal Server Error) جلوگیری شود.

۳. اعتبارسنجی لبه سیستم (Edge Validation) با Pydantic

داده‌های ورودی کاربر غیرقابل اعتماد هستند. با استفاده از Schemaهای سخت‌گیرانه Pydantic در لایه روتر FastAPI، داده‌ها قبل از ورود به چرخه پیش‌بینی، اعتبارسنجی شده (Type Checking & Range Validation) و امنیت سیستم تضمین می‌گردد.

۴. غلبه بر نفرین ابعاد (Curse of Dimensionality)

در بنچمارک الگوریتم‌ها، مدل پیچیده‌تر (Random Forest) با خطای بالاتری نسبت به Linear Regression مواجه شد. به دلیل استفاده از One-Hot Encoding، داده‌ها تبدیل به ماتریس‌های خلوت (Sparse Matrices) شدند. رگرسیون خطی با اختصاص وزن مستقیم، اصل Occam's Razor را اثبات کرد و به عنوان Champion Model انتخاب شد.

📊 نتایج بنچمارک (A/B Testing - 5-Fold CV)

Algorithm

Mean R² Score

RMSE (USD)

Status

Linear Regression

0.5253

$52,532

🏆 Champion (Deployed)

Random Forest

0.4524

$57,302

Baseline

🛠️ پشته تکنولوژی (Tech Stack)

Serving & API: FastAPI, Uvicorn, Pydantic

Machine Learning: Scikit-Learn, NumPy, Joblib (Optimized Serialization)

Data Persistence: Pandas, SQLAlchemy, SQLite

Design Patterns: Dependency Injection, Orchestrator, Factory Pattern

🚀 راهنمای نصب و اجرا (Quick Start)

۱. نصب نیازمندی‌ها:

```bash
pip install -r requirements.txt
```

۲. اجرای ارکستراتور (ETL + Training + Serving):
با اجرای دستور زیر، سیستم به طور خودکار پایپ‌لاین را در صورت عدم وجود Artifactها ران کرده و سرور را در localhost:8000 بالا می‌آورد:

```bash
python main.py
```

👉 رابط کاربری تعاملی (Swagger): http://127.0.0.1:8000/docs

🛣️ نقشه راه توسعه (Roadmap to MLOps Level 1)

برای ارتقای این MVP به یک پلتفرم کاملاً بالغ، موارد زیر در دستور کار قرار دارند:

[ ] کانتینرسازی محیط با Docker و docker-compose.

[ ] پیاده‌سازی تست‌های خودکار (Unit/Integration Tests) با Pytest.

[ ] اضافه کردن MLflow برای Model Registry و Experiment Tracking.

[ ] راه‌اندازی CI/CD Pipeline با GitHub Actions.

توسعه‌دهنده: متین محمدی - مهندس نرم‌افزار