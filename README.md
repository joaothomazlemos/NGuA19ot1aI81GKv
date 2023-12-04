Background:

"Our small startup specializes in delivering machine learning solutions tailored for the European banking sector. Our focus spans diverse challenges, encompassing fraud detection, sentiment classification, and customer intention prediction. Currently, our primary objective is to construct a resilient machine-learning system that harnesses insights from call center data. The ultimate aim is to enhance the success rate of customer calls related to our client's products. We are actively engaged in the design and development of a dynamic machine learning product, aiming for both high success outcomes and interpretability. This ensures our clients can make informed decisions based on the generated insights."

Data Description:

The data comes from the direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case, a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

Attributes:

age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

Output (desired target):

y - has the client subscribed to a term deposit? (binary)

## Goal(s):

Predict if the customer will subscribe (yes/no) to a term deposit (variable y)

## Conclusion
Random forest and XGBClassifier were the best, because the performe well both on inbalanced test data as well as balanced training data. ALso, the train and test socres are similar, with train score a little higher as espected (?).

Finally, they beat the scoreline proposed for this project, which was 81%.
