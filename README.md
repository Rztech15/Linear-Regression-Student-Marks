# 🎯 Linear Regression: Student Marks Prediction

Predict **student marks** based on **study hours** and **attendance** using **Gradient Descent**.

---

## 📊 Dataset

| Study (x₁) | Attendance (x₂) | Marks (y) |
|------------|----------------|------------|
| 2          | 60             | 50         |
| 4          | 70             | 65         |
| 6          | 80             | 80         |
| 8          | 90             | 95         |

> ✅ Goal: Build a regression model to predict marks for new students.

---

## Step 1️⃣: Hypothesis Function

The linear regression model:

\[
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2
\]

- \(x_1\) = Study hours  
- \(x_2\) = Attendance  
- θ₀ = Intercept  
- θ₁, θ₂ = Coefficients  

**Initial values:**

\[
\theta_0 = 0, \quad \theta_1 = 0, \quad \theta_2 = 0
\]

**Predicted marks initially:**

| Student | hθ(x) |
|---------|-------|
| 1       | 0     |
| 2       | 0     |
| 3       | 0     |
| 4       | 0     |

> 💡 All predictions are zero at the start.

---

## Step 2️⃣: Calculate Errors

\[
Error = h_\theta(x) - y
\]

| Student | hθ(x) | y | Error |
|---------|-------|---|-------|
| 1       | 0     | 50  | -50   |
| 2       | 0     | 65  | -65   |
| 3       | 0     | 80  | -80   |
| 4       | 0     | 95  | -95   |

> 💡 Errors show the difference between predicted and actual marks.

---

## Step 3️⃣: Compute Gradients (using 2/m)

\[
\frac{2}{m} = \frac{2}{4} = 0.5
\]

### Gradient for θ₀

\[
\sum(h-y) = -50-65-80-95=-290
\]

\[
Gradient = 0.5 \times (-290) = -145
\]

### Gradient for θ₁

\[
\sum(h-y)x_1 = (-50*2)+(-65*4)+(-80*6)+(-95*8)=-1600
\]

\[
Gradient = 0.5 \times (-1600) = -800
\]

### Gradient for θ₂

\[
\sum(h-y)x_2 = (-50*60)+(-65*70)+(-80*80)+(-95*90)=-22500
\]

\[
Gradient = 0.5 \times (-22500) = -11250
\]

---

## Step 4️⃣: Update Parameters

Update rule:

\[
\theta_j := \theta_j - \alpha (\text{gradient})
\]

- Learning rate: α = 0.01

| Parameter | Old Value | Gradient | New Value |
|-----------|-----------|---------|-----------|
| θ₀        | 0         | -145    | 1.45      |
| θ₁        | 0         | -800    | 8         |
| θ₂        | 0         | -11250  | 112.5     |

> 💡 Parameters updated after first iteration. Gradient Descent repeats until error is minimized.

---

## Step 5️⃣: Regression Model (After First Iteration)

\[
\text{Marks} = 1.45 + 8(\text{Study}) + 112.5(\text{Attendance})
\]

> ⚠️ Observation: Attendance values are much larger than study hours. Feature scaling is recommended in real applications.

---

## Step 6️⃣: Predict New Student (Optional)

For Study = 5, Attendance = 75:

\[
Marks = 1.45 + 8(5) + 112.5(75) \approx 8478.95
\]

> ⚠️ Exaggerated due to unscaled data.

---

## Step 7️⃣: Optional Python Code

```python
import numpy as np

# Dataset
X = np.array([[2,60],[4,70],[6,80],[8,90]])
y = np.array([50,65,80,95])
m = len(y)

# Initial parameters
theta = np.array([0.0,0.0,0.0])
alpha = 0.01
iterations = 1  # First iteration example

# Add intercept term
X_b = np.c_[np.ones((m,1)), X]
import numpy as np
from colorama import Fore, Style, init

# 🎨 Initialize colorama
init(autoreset=True)

# Dataset
X = np.array([[2,60],[4,70],[6,80],[8,90]])
y = np.array([50,65,80,95])
m = len(y)

# Initial parameters
theta = np.array([0.0,0.0,0.0])
alpha = 0.01
iterations = 1  # First iteration example

# Add intercept term
X_b = np.c_[np.ones((m,1)), X]

for _ in range(iterations):
    h = X_b.dot(theta)  # 🔹 Predicted values
    gradient = (2/m)*X_b.T.dot(h - y)  # 🔹 Compute gradient
    theta = theta - alpha*gradient  # 🔹 Update parameters

# ------------------- Colorful Outputs -------------------

# ✅ Updated parameters in green
print(Fore.GREEN + "✅ Updated Parameters (θ₀, θ₁, θ₂):" + Style.RESET_ALL, theta)

# 📊 Predictions in cyan
predictions = X_b.dot(theta)
print(Fore.CYAN + "📊 Predictions for existing students:" + Style.RESET_ALL, predictions)

# 📝 Predict new student
study_new, attendance_new = 5, 75
marks_new = theta[0] + theta[1]*study_new + theta[2]*attendance_new
print(Fore.MAGENTA + f"📝 Predicted Marks for Study={study_new}h & Attendance={attendance_new}%:" + Style.RESET_ALL, marks_new)

# ❌ Show errors for each student in red
errors = predictions - y
for i, err in enumerate(errors, start=1):
    print(Fore.RED + f"❌ Student {i} Error:" + Style.RESET_ALL, err)

# 💡 Optional: Highlight observation
print(Fore.YELLOW + "💡 Note: Attendance has larger scale than Study hours. Feature scaling recommended in real applications." + Style.RESET_ALL)

for _ in range(iterations):
    h = X_b.dot(theta)
    gradient = (2/m)*X_b.T.dot(h - y)
    theta = theta - alpha*gradient

print("Updated parameters:", theta)
print("Predictions:", X_b.dot(theta))
