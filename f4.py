from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, unix_timestamp
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer, PolynomialExpansion
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm vẽ biểu đồ histogram cho 1 cột sau khi chuyển về Pandas DataFrame
def plot_histogram(pdf, col_name, title="Histogram"):
    plt.figure(figsize=(8, 4))
    sns.histplot(pdf[col_name].dropna(), kde=True)
    plt.title(f"{title}: {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.show()

# Hàm in thống kê mô tả của cột (sử dụng Pandas)
def explore_column(pdf, col_name):
    print(f"Thống kê mô tả cho {col_name}:")
    print(pdf[col_name].describe())
    plot_histogram(pdf, col_name)

# Tạo Spark Session
spark = SparkSession.builder.appName("BigDataProject").getOrCreate()

# --- 1. Đọc dữ liệu ---
df_train_17 = spark.read.csv("C:/Nam_Data/bigdata/BTL/zillow-prize-1/train_2017.csv", header=True, inferSchema=True)
df_prop_17  = spark.read.csv("C:/Nam_Data/bigdata/BTL/zillow-prize-1/properties_2017.csv", header=True, inferSchema=True)
df_train_16 = spark.read.csv("C:/Nam_Data/bigdata/BTL/zillow-prize-1/train_2016_v2.csv", header=True, inferSchema=True)
df_prop_16  = spark.read.csv("C:/Nam_Data/bigdata/BTL/zillow-prize-1/properties_2016.csv", header=True, inferSchema=True)
df_test     = spark.read.csv("C:/Nam_Data/bigdata/BTL/zillow-prize-1/sample_submission.csv", header=True, inferSchema=True)

# Merge dữ liệu
df16 = df_train_16.join(df_prop_16, "parcelid", "left")
df17 = df_train_17.join(df_prop_17, "parcelid", "left")
train = df16.union(df17)
print("Schema sau khi merge:")
train.printSchema()

# Lấy mẫu nhỏ để khám phá dữ liệu
sample_pdf = train.sample(False, 0.01, seed=42).toPandas()
print("Thống kê mô tả ban đầu:")
print(sample_pdf.describe())

# --- 2. Xử lý giá trị thiếu ---
numeric_cols = [c for c, dtype in train.dtypes if dtype in ['int', 'double']]
imputer = Imputer(inputCols=numeric_cols, outputCols=[f"{c}_imputed" for c in numeric_cols])
train = imputer.fit(train).transform(train)
for c in numeric_cols:
    train = train.drop(c).withColumnRenamed(f"{c}_imputed", c)

# Với các cột kiểu date, chuyển sang UNIX timestamp nếu có
date_cols = [c for c, dtype in train.dtypes if dtype == "date"]
for c in date_cols:
    train = train.withColumn(c, unix_timestamp(col(c)))

# Khám phá một vài cột số sau imputation (lấy mẫu về Pandas)
sample_pdf = train.sample(False, 0.01, seed=42).toPandas()
for col_name in numeric_cols[:3]:
    explore_column(sample_pdf, col_name)

# --- 3. Xử lý ngoại lệ (Outlier Detection) ---
def remove_outliers_iqr(df, col_name):
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    if len(quantiles) < 2:
        return df
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

for c in numeric_cols:
    train = remove_outliers_iqr(train, c)

print("Sau xử lý ngoại lệ:")
train.printSchema()
# Khám phá sau khi loại bỏ ngoại lệ
sample_pdf = train.sample(False, 0.01, seed=42).toPandas()
for col_name in numeric_cols[:3]:
    explore_column(sample_pdf, col_name)

# --- 4. Xử lý biến phân loại ---
categorical_features = [c for c, dtype in train.dtypes if dtype == 'string']
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="skip") for c in categorical_features]

# --- 5. Feature Engineering ---
# Ví dụ: tạo đặc trưng tương tác giữa 2 cột số (nếu có đủ cột)
if len(numeric_cols) >= 2:
    interaction_col = numeric_cols[0] + "_" + numeric_cols[1] + "_interact"
    train = train.withColumn(interaction_col, col(numeric_cols[0]) * col(numeric_cols[1]))
    numeric_cols.append(interaction_col)
    print(f"Đã tạo đặc trưng tương tác: {interaction_col}")

# Tạo đặc trưng đa thức bậc 2
assembler_poly = VectorAssembler(inputCols=numeric_cols, outputCol="num_features_vector")
train = assembler_poly.transform(train)
polyExpansion = PolynomialExpansion(degree=2, inputCol="num_features_vector", outputCol="poly_features")
train = polyExpansion.transform(train)
print("Đã tạo đặc trưng đa thức bậc 2.")

# --- 6. Feature Selection dựa trên Variance ---
variance_df = train.select(*[F.variance(col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()
selected_numeric = [c for c, var in variance_df.items() if var is not None and var >= 0.01]
numeric_cols = selected_numeric  # Cập nhật lại danh sách
print("Các cột số được chọn sau feature selection (dựa trên variance):", numeric_cols)

# --- 7. Chuẩn hóa dữ liệu ---
assembler = VectorAssembler(inputCols=numeric_cols + [c + "_index" for c in categorical_features], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# --- 8. Xây dựng Pipeline ---
pipeline_stages = indexers + [assembler, scaler]
pipeline = Pipeline(stages=pipeline_stages)
pipeline_model = pipeline.fit(train)
train = pipeline_model.transform(train)

# Khám phá vector đặc trưng sau chuẩn hóa (lấy mẫu về Pandas)
sample_pdf = train.select("scaled_features").sample(False, 0.01, seed=42).toPandas()
print("Một vài giá trị của scaled_features:")
print(sample_pdf.head())

# --- 9. Chia tập dữ liệu ---
train_data, test_data = train.randomSplit([0.8, 0.2], seed=42)

# --- 10. Huấn luyện và đánh giá mô hình ---
# Định nghĩa hàm đánh giá với nhiều chỉ số
def evaluate_model(predictions, labelCol="logerror"):
    evaluator_rmse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse")
    evaluator_mse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mse")
    evaluator_mae = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mae")
    evaluator_r2  = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="r2")
    rmse = evaluator_rmse.evaluate(predictions)
    mse  = evaluator_mse.evaluate(predictions)
    mae  = evaluator_mae.evaluate(predictions)
    r2   = evaluator_r2.evaluate(predictions)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

# Huấn luyện mô hình RandomForestRegressor
rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="logerror")
rf_model = rf.fit(train_data)
y_pred_rf = rf_model.transform(test_data)
print("Kết quả đánh giá mô hình Random Forest:")
evaluate_model(y_pred_rf)

# Huấn luyện mô hình Gradient Boosting (GBT)
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="logerror", maxIter=100)
gbt_model = gbt.fit(train_data)
y_pred_gbt = gbt_model.transform(test_data)
print("Kết quả đánh giá mô hình Gradient Boosting:")
evaluate_model(y_pred_gbt)

# Vẽ biểu đồ phân bố của dự đoán so với giá trị thực (chuyển mẫu về Pandas)
def plot_prediction_distribution(predictions_df, labelCol="logerror"):
    pdf = predictions_df.select(labelCol, "prediction").toPandas()
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pdf[labelCol], y=pdf["prediction"], alpha=0.5)
    plt.xlabel("Giá trị thực")
    plt.ylabel("Dự đoán")
    plt.title("Biểu đồ so sánh giá trị thực và dự đoán")
    plt.show()

plot_prediction_distribution(y_pred_rf)
plot_prediction_distribution(y_pred_gbt)
