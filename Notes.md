1.Hum kisi ek store ka sales ko analyze karke predict karenge ki future me kitna revenue generate hoga. ye Business and Marketing decisions lene me helpful hoga.

2.what is retail chain? Example->walmart,DMart,Reliance Fresh
Retail chain ek company hai jo apne products ko multiple store me sell karta hai. 
Inke store multiple location me hote h.
Ek centralized management system ke through manage kiya jata h.
In stores me haar jagah same product and same offer milta h.

# Problem statement
Big Mart ek retail chain.  Haar stores ka sales different hota h because ,locations,customer demogrpahics(customer ki pasand) , demand of product haar jagah alag alag hota h.
Company ye predict karna chahti hai ki,unke haar stores ka sales performance kaisa hoga.Taki wo apne marketing and inventory ko optimize kar sake.

I have to find out these insights:-
i.kon sa product sabse jyada sell hua h.
ii.kon sa location ka sales sabse high and kiska sales sabse low hua h.
iii.seasonal trends kya h.  ( Diwali ke time mithai aur decoration items zyada bikte hain. )
iv.customer behaviour kaisa h (Price-Sensitive Customers,Quality-Conscious Customers,Trend-Followers) 
(customer behaviour unke prefrences ,budget pe depend karta h)
v.koi product jyada sell hua h to kyu jyada sell hua and koi product kum sell hua h to kyu kum sell hua (isse company decide karega ki kis area me wo product apne inventory  kitna amount me rakhna h)

# objective 
Item_Outlet_Sales->dependent and all other column -> independent

My Task => ek regression model banane ka hai jisme aapko saare features (jaise Item_Weight, Item_Fat_Content, Item_MRP, etc.) ke basis par Item_Outlet_Sales ko predict karne ke liye model train karna hai.

Scenario:
i.Training Phase: Aapke paas ek dataset hai, jisme har item ke liye Item_Outlet_Sales diya gaya hai. Aapko iske features (jaise Item_Weight, Item_MRP, Outlet_Size, etc.) ko dekhte hue Item_Outlet_Sales ko predict karna hoga.

ii.Prediction Phase: Jab model trained ho jaayega, toh aap naye data points (new items) ke liye Item_Outlet_Sales predict kar sakte ho.

Note:mujhe ek model banana hai jisme pehle mujhe,given Item_Outlet_Sales data ko train krna h ,and fir agar new data aayega to uska Item_Outlet_Sales predict karna h.

# work flow
i.Data collection (Dataset are in csv format containes price,sales..)
ii.Data cleaning (Handle missing values,Remove duplicates,)
iii.EDA (Exploratory Data Analysis) ->Analysis and visualization  of the data ,jisse hum data ka patterns and relationship samjh dake.
iv.Feature Engineering
v.Train-Test split -> 80% data Training ke liye and 20% data Testing ke liye. 
vi.Model Training->Linear Regression model ko training data ke upar fit karta h.
iss step me hum model ko Build karenge.
vii.Model prediction->Jab model train ho jata h to ,testing data ke liye prediction kiya jata h.
viii.Model Evaluation->Evaluating performance of model using Mean squared Error(MSE) and R-squared(R^2)
(iss step me model testing hoga)
MSE->MSE se model ka accuracy find karte h and R^2 se model ka fitness check karte h.

Test the model's performance using various metrics:-
For classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
For regression: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).

ix.Result Analysis 
x.Optimzation->if model accuracy is low then, Different algorithm try karke accuracy increase karte h (like Random forest,XGBoost..)

# EDA ?
EDA is a process ,jisme hum data ko analyze karte hai taki uska patters,trends,outliers and relationship samjh sake.
Data Cleaning,Data Visualization,Outliers Detection,Handling Missing Data ,Data Summary

# solution
solve this problem using Regression ,because here values are numeric and continuous (example->sales,age,price)

# Nominal vs Ordinal Data
Types of categorical data:-Nominal and Ordinal Data

categorical data->wo data jo categories me divide hota h.Categorical data ki values usually text ya labels hoti hain, jaise "Red", "Male", "Yes".
Isme values numerical form me nahi hoti . Ye data quantitative nahi hota, matlab ispar directly mathematical operations (addition, subtraction) nahi kar sakte.

Nominal Data: Koi order nahi hota (e.g., Colors: Red, Green, Blue).
Ordinal Data: Order hota hai, but difference uniform nahi hota (e.g., Ratings: Poor < Average < Good).

Machine Learning models numerical values ka use karta h, isliye categorical data ko encode karna padta hai (e.g., One-Hot Encoding, Label Encoding,ordinal encoding).

# Types of Emcoding->
Encoding ka kaam categorical data ko numerical form me convert karna hota hai, taki machine learning models ise samajh saken. 

i.Label Encoding : Each  unique  category ko ek unique integer value assign karte hai.
Example: Red = 0, Green = 1, Blue = 2,Red   -> 0  ,Blue  -> 2

when to use->Jab categories ke beech koi order na ho (jaise color ya city ke naam).

ii.One-Hot Encoding:Har category ke liye ek separate binary column banata hai.
Color: Red, Green, Blue
Red = [1, 0, 0]
Green = [0, 1, 0]
Blue = [0, 0, 1]

when to use->Jab koi order na ho aur aap kisi category ko preference dena nahi chahte

iii.Ordinal Encoding:categories ko order wise number assign karte h.
Example:High School -> 1  ,Bachelor -> 2  , Master-> 3  , PhD-> 4  

when to use->Jab categories me ek logical order ho (jaise level ya rank).

# In this project we use label encoding?
Because Label encoding is best for Tree-based models(XGBOOST,RANDOM FOREST).
and One-Hot encoding is best for Distance based model (KNN,Linear regression,Logistic Regression)

# R² Score 
R² Score ka use regression models ka accuracy find karne ke liye hota h. 
(average value se kitna aacha perform kar rha h model, agar negative aaya ,
to average value se bhi kharab predict kar rha h and model kharab h)

R²=1 :Perfect fit ,  R²=0 :no prediction  ,  R²<0 :worst performance

R² Score Kab Use Karna Chahiye? 
Regression Problems: Jab target variable continuous ho (e.g., house prices, sales prediction).

=>  R²=0.75 :  75% sales ki variability model explain kar raha hai,
               aur 25% variability unexplained hai (errors aur unknown factors ki wajah se).


# Install amd Activate env
i.python -m venv myenv
ii.myenv\Scripts\activate

# Install libraries 
pip install numpy pandas seaborn matplotlib scikit-learn xgboost

# Note: 
 i. starting me " r " laga do bas
 pd.read_csv(r"C:\Users\sagar\Desktop\data analytics all project\Big Mart sales prediction\dataset\Train.csv")

or
ii. \ ye slash ko hata ke / ye slash daal do
pd.read_csv("C:/Users/sagar/Desktop/data analytics all project/Big Mart sales prediction/dataset/Train.csv")

# understand Columns
Item_Identifier	->uniquely identify the product (FD-food , DR->drinkable ,  NC->non-consumeable)

Item_Visibility-> ek item kitna visible hai customers ke liye.
suppose koi item store ke front me h  to uska sales jyada hota h.

Outlet_Identifier->haar outlet ko uniquely identify karne ke liye.

