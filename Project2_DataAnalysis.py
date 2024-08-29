#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing liabriries 
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


#reading the data
df = pd.read_csv(r"C:\Users\hala mohamed\Downloads\fifa_eda_stats.csv")


# In[3]:


#knowing shape of data
df.shape


# In[4]:


#basic explortaion
df.info()


# In[5]:


#see sample of the data
df.head()


# In[6]:


#know columns which have null values 
df.isnull().sum()
#conclusion:club,joined,loaned from,contract valid until,release clause columns miss alot of data


# In[7]:


#cleaning data
#filling the missing values for the continous variables for proper data visualization
df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)
df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)
df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)
df['Curve'].fillna(df['Curve'].mean(), inplace = True)
df['FKAccuracy'].fillna(df['FKAccuracy'].mean(), inplace = True)
df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)
df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)
 

# drop unuseful columns
columns_to_drop = ['Loaned From', 'Jersey Number']   
df = df.drop(columns=columns_to_drop)


# In[8]:


#know which columns are catgorical and which are countionus
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

print("Numerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)


# In[9]:


#we need to convert height and weight to numerical values, to be fill with mean
# Strip the "lbs" suffix and convert to an integer
def convert_weight(x):
    if isinstance(x, str):  # Ensure x is a string
        if x[-3:] == "lbs":
            return round(float(x[:-3]) * 0.453592, 2)
        elif x[-2:] == "kg":
            return float(x[:-2])
    return x  # Return original value if not a string

df['Weight'] = df['Weight'].apply(convert_weight)

def convert_height_to_inches(height_str):
    """Converts height in feet and inches format to inches (integer)."""
    if isinstance(height_str, str):
        feet, inches = height_str.split("'")
        total_inches = int(feet) * 12 + int(inches)
        return total_inches
    else:
        return height_str  


# Apply the function to the Height column
df['Height'] = df['Height'].apply(convert_height_to_inches)
 

 

 


# In[10]:


# Fill missing numerical values with the mean of their respective columns
df.fillna(df.mean(), inplace=True)



# In[11]:


#check the validity of the step
df.isnull().sum()


# In[12]:


#fill the categorical columns with mode
df.fillna(df.mode().iloc[0], inplace=True)


# In[13]:


#verify the previous step
df.isnull().sum()


# In[14]:


#Here , we have Finished the Data Cleaning phase


# In[15]:


#here i made some functions that are intended to calculate different aggregate scores
#for players based on various attributes in the FIFA dataset, they will be useful for further exploration
def defending(data):
    '''Purpose: Calculates an overall "defending" score for a player.
       Attributes Considered: Marking, StandingTackle, SlidingTackle
       Calculation:
               First, it calculates the mean of Marking, StandingTackle, and SlidingTackle.
               Then, it takes the mean of the resulting average.
               Finally, it rounds this average and converts it to an integer'''
    
    return int(round(data[['Marking', 'StandingTackle', 'SlidingTackle']].mean().mean()))
def general(data):
    '''Purpose: Computes a "general" skills score.
       Attributes Considered: HeadingAccuracy, Dribbling, Curve, BallControl'''
    
    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 'BallControl']].mean()).mean()))

def mental(data):
    '''Purpose: Computes a "mental" abilities score.
        Attributes Considered: Aggression, Interceptions, Positioning, Vision, Composure'''
    
    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 'Vision', 'Composure']].mean()).mean()))

def passing(data):
    '''Purpose: Calculates a "passing" skills score.
       Attributes Considered: Crossing, ShortPassing, LongPassing'''
    
    return int(round((data[['Crossing', 'ShortPassing', 'LongPassing']].mean()).mean()))

def mobility(data):
    '''Purpose: Calculates a "mobility" score.
      Attributes Considered: Acceleration, SprintSpeed, Agility, Reactions'''
    
    return int(round((data[['Acceleration', 'SprintSpeed', 'Agility', 'Reactions']].mean()).mean()))

def power(data):
    '''Purpose: Computes a "power" score.
       Attributes Considered: Balance, Jumping, Stamina, Strength'''
    return int(round((data[['Balance', 'Jumping', 'Stamina', 'Strength']].mean()).mean()))

def rating(data):
    '''Purpose: Calculates a combined "rating" score based on overall and potential ratings.
       Attributes Considered: Potential, Overall'''
    return int(round((data[['Potential', 'Overall']].mean()).mean()))

def shooting(data):
    '''Purpose: Computes a "shooting" skills score.
       Attributes Considered: Finishing, Volleys, FKAccuracy, ShotPower, LongShots, Penalties'''
    
    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 'ShotPower', 'LongShots', 'Penalties']].mean()).mean()))


# In[16]:


#Adding these to data
df['Defending'] = df.apply(defending, axis = 1)
df['General'] = df.apply(general, axis = 1)
df['Mental'] = df.apply(mental, axis = 1)
df['Passing'] = df.apply(passing, axis = 1)
df['Mobility'] = df.apply(mobility, axis = 1)
df['Power'] = df.apply(power, axis = 1)
df['Rating'] = df.apply(rating, axis = 1)
df['Shooting'] = df.apply(shooting, axis = 1)


# In[17]:


#we can create a data frame containing the important information  that may help for the analysis
players = df[['Name', 'Defending', 'General', 'Mental', 'Passing', 
                'Mobility', 'Power', 'Rating', 'Shooting', 'Age',
                'Nationality',  'Club']]
players


# In[18]:


df.columns


# In[19]:


#After finishing the data cleaning phase and making extra step ti facilitate the analysis , i started the univarinet analysis 
#for some important features


# In[20]:


x = df['Age']
plt.figure(figsize = (12, 8))
plt.style.use('ggplot')
ax = sb.histplot(x, bins = 20, kde = True, color='g')
ax.set_xlabel(xlabel = 'Age of the Players', fontsize = 16)
ax.set_title(label = 'Histogram for Age distribution of Players', fontsize = 20)
plt.show()


# In[21]:


# to spot the outlires in Age ,we need to view the boxplot
plt.figure(figsize=(10, 6))   
sb.boxplot(x=df['Age'], color='skyblue')

 
plt.title('Age Distribution of Players')
plt.xlabel('Age')
plt.ylabel('Frequency')

 
plt.show()


# In[22]:


#as you saw ,we spot some outlires that need to be removed
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

 
df = remove_outliers_iqr(df, 'Age')


# In[23]:


#to verify the step:
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
sb.boxplot(x=df['Age'], color='skyblue')

# Add titles and labels
plt.title('Age Distribution of Players')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[24]:


plt.figure(figsize=(10, 6))   
sb.histplot(df['Overall'], kde=True, bins=20, color='blue')

 
plt.title('Overall Rating Distribution of Players')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[25]:


#spot outliers 
plt.figure(figsize=(8, 6))   
sb.boxplot(x=df['Overall'], color='lightblue')

 
plt.title('Boxplot of Overall Rating')
plt.xlabel('Overall Rating')


# In[26]:


#remove outliers 

df = remove_outliers_iqr(df, 'Overall')


# In[27]:


#same for the rest of the columns
plt.figure(figsize=(8, 6))   
sb.boxplot(x=df['Potential'], color='lightblue')

 
plt.title('Boxplot of Potential Rating')
plt.xlabel('Potential Rating')


# In[28]:


#remove outliers 
df = remove_outliers_iqr(df, 'Potential')


# In[29]:


plt.figure(figsize=(8, 6))   
sb.boxplot(x=df['Potential'], color='lightblue')

 
plt.title('Boxplot of Potential Rating')
plt.xlabel('Potential Rating')


# In[30]:


df = remove_outliers_iqr(df, 'Height')


# In[31]:


print(df['Height'].describe())


# In[32]:


plt.figure(figsize=(8, 6))   
sb.boxplot(data=df,x='Height', color='lightblue')

 
plt.title('Boxplot of Height')
plt.xlabel('Height  Rating')
plt.show()


# In[33]:


plt.figure(figsize=(8, 6))   
sb.boxplot(data=df,x='Weight', color='lightblue',whis=2.0)

 
plt.title('Boxplot of Weight ')
plt.xlabel('Weight   Rating')


# In[34]:


#remove outlires 
df = remove_outliers_iqr(df, 'Weight')


# In[35]:


#first we need to remove k,m or similer values in columns Wage and value ,for making the analyis, and remving outliers 
# we need to build function to change its data type to float
def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x;
    if 'K' in x:
        if len(x) > 1:
            return float (x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float (x.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in x:
        if len(x) > 1:
            return float (x.replace('B', '')) * 1000000000
        return 1000000000.0
#apply the function to the columns
wage = df["Wage"].replace('[\€,]', "", regex=True).apply(value_to_float)
value = df["Value"].replace('[\€,]', "", regex=True).apply(value_to_float)
re=df["Release Clause"].replace('[\€,]', "", regex=True).apply(value_to_float)
df["Wage"] = wage
df["Value"] = value
df['Release Clause']=re


# In[36]:


#removing outlires
df = remove_outliers_iqr(df, 'Wage')
df = remove_outliers_iqr(df, 'Value') 


# In[37]:


sb.boxplot(data=df,x='Value', color='lightblue',whis=3.0)

 
plt.title('Boxplot of value ')
plt.xlabel('Value Rating')


# In[38]:


sb.boxplot(data=df,x='Wage', color='lightblue',whis=4.0)
plt.title('Boxplot of Wage ')
plt.xlabel('Wage Rating')


# In[39]:


df = remove_outliers_iqr(df, 'SprintSpeed')


# In[40]:


sb.boxplot(data=df,x='SprintSpeed', color='lightblue',whis=2.0)

 
plt.title('Boxplot of SprintSpeed ')
plt.xlabel('SprintSpeed Rating')


# In[41]:


def remove_outliers_from_columns(df, columns):
    """Removes outliers from multiple columns in a DataFrame."""
    df_copy = df.copy()  # Create a copy to avoid modifying the original
    for column in columns:
        # Calculate IQR for the column
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define upper and lower bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    return df_copy

# Columns to remove outliers from
numerical_columns = ['Defending', 'General', 'Mental', 'Passing', 
                     'Mobility', 'Power', 'Rating', 'Shooting']

# Remove outliers
players_no_outliers = remove_outliers_from_columns(players, numerical_columns)


# In[42]:


#after removing outlires from the important features we can start our analysis 


# In[43]:


descriptive_stats = players[['Defending', 'General', 'Mental', 'Passing', 
                            'Mobility', 'Power', 'Rating', 'Shooting', 'Age']].describe()
print(descriptive_stats)


# In[44]:


#visualize the distribution of numerical
for column in ['Defending', 'General', 'Mental', 'Passing', 'Mobility', 'Power', 'Rating', 'Shooting', 'Age']:
    plt.figure()  # Create a new figure for each plot
    sb.histplot(players[column], kde=True)   
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[45]:


#boxplot to verify the step of removing outlires
for column in numerical_columns:
    plt.figure()   
    sb.boxplot(data=players_no_outliers, x=column)
    plt.title(f'Box Plot of {column} ')
    plt.xlabel(column)
    plt.show()


# In[46]:


count = df["Nationality"].value_counts()
print(count)
players_country = count.idxmax()
num_players = count.max()
print("The country with the most number of players is", players_country, "with", num_players, "players.")


# In[47]:


plt.figure(figsize=(20,50))
sb.countplot(y="Nationality",data=df,palette="hsv")
plt.show()
#We can see that the most common nationality is England


# In[48]:


l=count.head()
plt.bar(l.index,l.values,color="blue")
plt.title('Top 5 Countries with the Most Number of Players')
plt.xlabel('Country')
plt.ylabel('Number of Players')
plt.figure(figsize=(12,6))
plt.show()


# In[ ]:





# In[49]:


#locate the player with hightest salary
salary=df.loc[df['Wage'].idxmax()]
print(salary)
s=salary["Wage"]
print(" highest salary is",s)


# In[50]:


plt.hist(df['Wage'], color='blue')
plt.title('Salary Range of Players')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[51]:


df['Height'] = df['Height'].astype(float)
tallest = df.loc[df['Height'].idxmax()]
print(tallest)
player_height = tallest['Height']
print("Height of the tallest player:",player_height)
 


# In[52]:


foot=df["Preferred Foot"].value_counts()
print(foot)


# In[53]:


#each foot preference using the value_counts() function, right foot is most preferred
plt.bar(foot.index,foot.values,color="blue")
plt.title('Preferred Foot of Players')
plt.xlabel('Preferred Foot')
plt.ylabel('Number of Players')
plt.show()


# In[54]:


#count the number of players for each club using the value_counts() function,
#then print the first 30 rows these club have most no of players . 
no_player=df["Club"].value_counts()
no_player.head(100)


# In[55]:


body_type_counts = df['Body Type'].value_counts()

# Plot the distribution
plt.figure(figsize=(10, 6))
sb.barplot(x=body_type_counts.index, y=body_type_counts.values, palette="viridis")
plt.title('Distribution of Body Types in FIFA 2019')
plt.xlabel('Body Type')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()
#conclusion :most frequnt body type is normal


# In[56]:


#different work rate of players
df['Work Rate'].value_counts()


# In[57]:


fig, ax = plt.subplots(figsize=(12,8))
graph = sb.countplot(ax=ax,x=df['Work Rate'], data=df, palette = 'PuBuGn_d')
graph.set_title('Work Rate of the Players', fontsize = 20)
graph.set_xticklabels(graph.get_xticklabels(), rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[58]:


#visualize the overall score of players 
x = df['Overall']
plt.figure(figsize=(18,10))
ax = sb.countplot(x=x, palette='rocket')
ax.set_xlabel(xlabel = "Player's Overall Scores", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of Players', fontsize = 16)
ax.set_title(label = 'Distribution of Players Overall Scores', fontsize = 20)
plt.show()


# In[59]:


# different positions acquired by the players 
plt.figure(figsize = (18, 8))
ax = sb.countplot(x='Position', data = df, palette = 'PuBuGn_d')
ax.set_xlabel(xlabel = 'Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Number of Players', fontsize = 16)
ax.set_title(label = 'Count of Players with Positions', fontsize = 20)   
plt.show()


# In[60]:


top_countries = df['Nationality'].value_counts().head(10)
top_countries_name = top_countries.index
df_country_age = df.loc[df['Nationality'].isin(top_countries_name) & df['Age']]
plt.figure(1 , figsize = (12,6))
sb.boxplot(x = 'Nationality' , y = 'Age' , data = df_country_age, palette='rocket')
plt.title('Age Distribution in top countries')
plt.xticks(rotation = 50)
plt.show()
#we can see that Geramany is the least range of ages


# In[61]:


#some famous clubs (from my prespective)
clubs = ['FC Barcelona','Real Madrid','Juventus','Liverpool','Manchester United',
         'Chelsea','Arsenal','Paris Saint-Germain' ,'FC Bayern München','Manchester City']
#Age distribution in famous clubs
df_club_age = df.loc[players['Club'].isin(clubs) & df['Age']]
plt.figure(1 , figsize = (12,6))
sb.boxplot(x = 'Club', y = 'Age' , data = df_club_age, palette='spring')
plt.title('Age Distribution in famous clubs')
plt.xticks(rotation = 50)
plt.show()
#observation: Overall, the age distribution of players across the clubs is relatively similar. The median age for most clubs falls between 20 and 22, with a few outliers.
#There is some variation in the spread of ages within each club. Some clubs have a wider range of player ages, while others have a more concentrated group of players around the median age.
#Real Madrid and Chelsea have the lowest median ages among the clubs. This suggests that these teams may have a younger squad on average compared to the others.


# In[62]:


#overall rating
df_club_rating = df.loc[df['Club'].isin(clubs) & df['Overall']]
plt.figure(1 , figsize = (12,6))
sb.boxplot(x = 'Club' , y = 'Overall' , data = df_club_rating, palette='PuBuGn_d')
plt.title('Overall Rating Distribution in famous clubs')
plt.xticks(rotation = 50)
plt.show()
#Paris Saint-Germain and FC Barcelona have the highest median ratings, suggesting they have a generally higher-rated squad


# In[63]:


df_sorted = df.sort_values(by='Overall', ascending=False)

# Get the top 10 players
top_10_players = df_sorted.head(10)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_10_players['Name'], top_10_players['Overall'], color='skyblue')
plt.xlabel('Player Name')
plt.ylabel('Overall Rating')
plt.title('Top 10 Players by Overall Rating (FIFA 2019)')
plt.xticks(rotation=45)
plt.show()
#we can see the Casillas is the highest overall score


# In[64]:


#highest Earners
df2 = df.sort_values(by='Wage', ascending=False)[['Name', 'Club', 'Nationality', 'Overall', 'Age', 'Wage']].head(5)

 
sb.barplot(x='Name', y='Wage', data=df2)

# Add plot labels and title
plt.xlabel('Player Name')
plt.ylabel('Wage (€)')
plt.title('Top 5 Highest-Paid Players')

 
plt.xticks(rotation=45)

 
plt.show()


# In[65]:


sb.pairplot(df[['Overall', 'Potential', 'Age', 'Value']])
plt.show()
#obervation:
#Overall vs. Potential: There's a strong positive correlation between Overall and Potential, as expected. Players with higher overall ratings generally have greater potential for growt
#Potential vs. Age: There seems to be a slight negative correlation between Potential and Age, suggesting that younger players generally have higher potential for growth.


# In[66]:


#check for duplicated values
df.duplicated().sum()
#no duplicated values to handle


# In[67]:


df['Joined'].head()


# In[68]:


#one-hot encoding 
categorical_columns = ['Name', 'Nationality', 'Club', 'Preferred Foot',
       'Work Rate', 'Body Type', 'Position', 'Joined', 'Contract Valid Until']
df_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[71]:


from sklearn.model_selection import train_test_split
#splitting data
# Select features (X) and target variable (y) - Market Value
X = df_encoded.drop('Value', axis=1) 
y = df_encoded['Value']

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[75]:


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(numerical_cols)


# In[76]:


from sklearn.preprocessing import StandardScaler


#feature scaling
scaler = StandardScaler()
numerical_cols = ['ID', 'Age', 'Overall', 'Potential', 'Wage',
       'International Reputation', 'Weak Foot', 'Skill Moves', 'Height',
       'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
       'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',
       'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
       'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes',
       'Release Clause', 'Defending', 'General', 'Mental', 'Passing',
       'Mobility', 'Power', 'Rating', 'Shooting']

# Fit the scaler on the training data and transform both training and testing data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
 

