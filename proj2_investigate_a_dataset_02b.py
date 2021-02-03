import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


## 1 - Data Wrangling (increase the available data quality)
## 1a - Discovering: understand what is in your data
## 1b - Structuring: organize the data (i.e. One column may become two) Make computation and analysis easier
## 1c - Cleaning: errors/outliers skew data?  Consisten names such as IL for Illinois. Remove nulls. Use STD format.
## 1d - Enriching: augment with new data to improve decision
## 1e - Validating: repetitive programming sequences to verify data consistency, quality, and security
## 1f - Publishing: prepare data for downstream data analytics use (document)

df_all = pd.read_csv('tmdb-movies.csv')

# 1a - discover
df_all.info()
df_all.head(2)
df_all.describe()

# remove columns that are not necessary
cols_not_needed = ['budget', 'homepage', 'overview', 'revenue', 'tagline']
df_all.drop(cols_not_needed, axis=1, inplace=True)

# find duplicates and drop them
print('duplicates=', df_all.duplicated().value_counts())
df_all.drop_duplicates(keep = 'first', inplace = True)

# check for nulls
df_all.isna().sum()

# there are 0s in budget and revenue, drop those entries
df = df_all
df = df[df['budget_adj']  !=0]
df = df[df['revenue_adj'] !=0]

# view dtypes
print(df.dtypes)

# fix release_date dtype
df['release_date'] = pd.to_datetime(df['release_date'])
print(df.dtypes)


## 2 - Exploratory Data Analysis (EDA)
## Compute statistics and create visualizations

# insert new cols : profit and net profit margin ratio (NPMR)
df.insert(df.shape[1], 'profit_adj', df['revenue_adj'] - df['budget_adj'])
df.insert(df.shape[1], 'npmr', df['profit_adj'] / df['budget_adj'] * 100)

## TIME SERIES

# Q. Which year has the highest release of movies?

movies_per_year = df['release_year'].value_counts()
print(movies_per_year.idxmax())

plt.figure() # Numpy Style (More control over bins)
binwidth = 1
bin_min = df['release_year'].min() - 0.5
bin_max = df['release_year'].max() + 0.5 + 1.0
bins = np.arange(bin_min, bin_max, 1.0)
plt.hist( df['release_year'].astype(float), bins=bins)
plt.title('Movie Releases per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)

plt.figure() # Pandas Style
bins = df['release_year'].max()-df['release_year'].min() + 1
df['release_year'].hist(bins=bins)
plt.title('Movie Releases per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)

# Q. How has movie revenue changed thru the years?
rev_tot_per_year = df.groupby('release_year')['revenue_adj'].sum()
movies_per_year  = df.groupby('release_year')['release_year'].count()
avg_mov_rev_per_year = rev_tot_per_year / movies_per_year
plt.figure()
plt.plot(avg_mov_rev_per_year)
plt.title('Average Movie Revenue thru the years')
plt.xlabel('year')
plt.ylabel('avg_mov_rev_per_year[$]')

# Q. What is the distribution of revenue?

plt.figure()
binwidth = 1
bin_min = df['revenue_adj'].min() - 0.5
bin_max = df['revenue_adj'].max() + 0.5 + 1.0
bins = np.arange(bin_min, bin_max, 100000000.0)
plt.hist( df['revenue_adj'].astype(float), bins=bins)
plt.axvline(df['revenue_adj'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(df['revenue_adj'].mean()*1.1, max_ylim*0.9, 'Mean in USD: {:,.0f}'.format(df['revenue_adj'].mean()))
plt.yscale('log')
plt.title('Movie Revenue Distribution')
plt.xlabel('revenue_adj[$]')
plt.ylabel('Count')
plt.grid(True)

# Q. What is the distribution of profit?

plt.figure()
binwidth = 1
bin_min = df['profit_adj'].min() - 0.5
bin_max = df['profit_adj'].max() + 0.5 + 1.0
bins = np.arange(bin_min, bin_max, 100000000.0)
plt.hist( df['profit_adj'].astype(float), bins=bins)
plt.axvline(df['profit_adj'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(df['profit_adj'].mean()*1.1, max_ylim*0.9, 'Mean in USD: {:,.0f}'.format(df['profit_adj'].mean()))
plt.yscale('log')
plt.title('Movie Profit Distribution')
plt.xlabel('profit_adj[$]')
plt.ylabel('Count')
plt.grid(True)


# Q. MOVIE RUNTIME - Average Runtime Of Movies From Year To Year?
rt_tot_per_year = df.groupby('release_year')['runtime'].sum()
movies_per_year = df.groupby('release_year')['release_year'].count()
avg_rt_per_year = rt_tot_per_year / movies_per_year
plt.figure()
plt.plot(avg_rt_per_year)
plt.title('Average Movie Runtime per Year')
plt.xlabel('year')
plt.ylabel('average movie runtime[min]')

# Q. MOVIE RUNTIME - Runtime distribution?
plt.figure()
binwidth = 1
bin_min = 0
bin_max = round(df['runtime'].max()/10) * 10
bins = np.arange(bin_min, bin_max, 10.0)
plt.hist( df['runtime'].astype(float), bins=bins)
plt.axvline(df['runtime'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(df['runtime'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(df['runtime'].mean()))
plt.title('Movie Runtime Distribution')
plt.xlabel('runtime[min]')
plt.ylabel('Count')
plt.grid(True)

# Q. MOVIE RUNTIME - Runtime distribution- Outliers?
plt.figure()
plt.boxplot(df['runtime'], vert=False)
plt.title('Movie Runtime - Outliers - Box Plot')
plt.xlabel('runtime[min]')


# Q. RELATIONSHIPS - Revenue Vs. Release Month
movies_by_month_number = df.groupby(df['release_date'].dt.month).count()['imdb_id']
movies_by_month_number.rename_axis('month', inplace=True)
movies_by_month_name   = df.groupby(df['release_date'].dt.month_name()).count()['imdb_id']
movies_by_month_name.rename_axis('month', inplace=True)

plt.figure()
movies_by_month_number.plot(kind='bar')
plt.title('Movie Releases per Month')
plt.xlabel('release_date Month')
plt.ylabel('Count')


# Q. What kinds of properties are associated with movies that have high revenues?

# RELATIONSHIPS - Revenue Vs. Runtime
plt.figure()
plt.scatter(df['runtime'], df['revenue_adj'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('runtime[min]')
plt.ylabel('revenue_adj[$]')

# RELATIONSHIPS - Revenue Vs. Vote Average
plt.figure()
plt.scatter(df['vote_average'], df['revenue_adj'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('vote_average')
plt.ylabel('revenue_adj[$]')

# RELATIONSHIPS - Revenue Vs. Vote Count
plt.figure()
plt.scatter(df['vote_count'], df['revenue_adj'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('vote_count')
plt.ylabel('revenue_adj[$]')
# trend
fit = np.polyfit(df['vote_count'], df['revenue_adj'], 1)
fit_fn = np.poly1d(fit)
x = [df['vote_count'].min(), df['vote_count'].max()]
y = [fit_fn(df['vote_count'].min()), fit_fn(df['vote_count'].max())]
plt.plot(x,y, color='red')


# Q.
# Top 10 - Popularity
# Top 10 - number of votes movies
# Top 10 - vote average

popularity_top10   = df.nlargest(10, 'popularity',   keep='first')
vote_count_top10   = df.nlargest(10, 'vote_count',   keep='first')
vote_average_top10 = df.nlargest(10, 'vote_average', keep='first')

plt.figure()
plt.bar(popularity_top10['original_title'], popularity_top10['popularity'])
plt.title('Top 10 Popular Movies')
plt.xlabel('original_title')
plt.ylabel('IMDb Popularity Rating')
plt.xticks(rotation='vertical')

plt.figure()
plt.bar(vote_count_top10['original_title'], vote_count_top10['vote_count'])
plt.title('Top 10 Highest Votes Movies')
plt.xlabel('original_title')
plt.ylabel('# Votes')
plt.xticks(rotation='vertical')

plt.figure()
plt.bar(vote_average_top10['original_title'], vote_average_top10['vote_average'])
plt.title('Top 10 Vote Average')
plt.xlabel('original_title')
plt.ylabel('Vote Average')
plt.xticks(rotation='vertical')


# Q.
# Top 10 shortest/longest runtime movies
# Top 10 lo/hi movie budgets
# Top 10 lo/hi movie revenues

longest_top10    = df.nlargest(10,  'runtime',   keep='first')
short_top10      = df.nsmallest(10, 'runtime',   keep='first')
budget_hi_top10  = df.nlargest(10,  'budget_adj', keep='first')
budget_lo_top10  = df.nsmallest(10, 'budget_adj', keep='first')
revenue_hi_top10 = df.nlargest(10,  'revenue_adj', keep='first')
revenue_lo_top10 = df.nsmallest(10, 'revenue_adj', keep='first')
# lo/hi movie revenue index (for reference)
revenue_lo_ix = df['revenue_adj'].idxmin()
revenue_hi_ix = df['revenue_adj'].idxmax()

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].bar(longest_top10['original_title'], longest_top10['runtime'])
ax[0].set_xlabel('original_title')
ax[0].set_ylabel('Runtime[min]')
ax[1].bar(short_top10['original_title'], short_top10['runtime'])
ax[1].set_xlabel('original_title')
ax[1].set_ylabel('Runtime[min]')
plt.draw() # needed to get the movie titles aligned correctly
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha='right')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha='right')
fig.suptitle('Top 10 - Longest and Shortest Movies')

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].bar(budget_hi_top10['original_title'], budget_hi_top10['budget_adj'])
ax[0].set_xlabel('original_title')
ax[0].set_ylabel('budget_adj[$]')
ax[1].bar(budget_lo_top10['original_title'], budget_lo_top10['budget_adj'])
ax[0].set_ylabel('budget_adj[$]')
plt.draw() # needed to get the movie titles aligned correctly
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha='right')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha='right')
fig.suptitle('Top 10 - Highest and Lowest Movie Budgets')

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].bar(revenue_hi_top10['original_title'], revenue_hi_top10['revenue_adj'])
ax[0].set_xlabel('original_title')
ax[0].set_ylabel('revenue_adj[$]')
ax[1].bar(revenue_lo_top10['original_title'], revenue_lo_top10['revenue_adj'])
ax[0].set_ylabel('revenue_adj[$]')
plt.draw() # needed to get the movie titles aligned correctly
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha='right')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha='right')
fig.suptitle('Top 10 - Highest and Lowest Movie Revenues')


# Q. 
# Top 10 Most Frequent star cast?
# Top 10 Most Frequent Directors?
# Top 10 Most Popular Genres?
# Top 10 Most popular keywords?
# Top 10 Most popular Production Companies?

# Wrangling - Certain columns,contain multiple values separated by pipe (|) characters.

def df_string_separate_count(df_col):
    df_col_all     = df_col.str.cat(sep = '|')
    df_col_series  = pd.Series(df_col_all.split('|'))
    df_col_count   = df_col_series.value_counts(ascending = False)
    return df_col_count

prop1 = ['cast', 'director', 'genres', 'keywords', 'production_companies']
prop1_count = []

for i in range(len(prop1)):
    df_prop = df[prop1[i]]
    prop1_count.append(df_string_separate_count(df_prop))

for i in range(len(prop1)):
    plt.figure()
    prop1_count[i][0:10].plot(kind='bar')
    plt.xlabel(prop1[i])
    plt.ylabel('Movie Count')
    plt.title('Top 10 ' + prop1[i])
    plt.grid(True) 
    

# Q. 
# Top 10 Most Frequent star cast?
# Top 10 Most Frequent Directors?
# Top 10 Most Popular Genres?
# Top 10 Most popular keywords?
# Top 10 Most popular Production Companies?

# Wrangling - Certain columns,contain multiple values separated by pipe (|) characters.

actor_all    = df['cast'].str.cat(sep = '|')
actor_series = pd.Series(actor_all.split('|'))
actor_count  = actor_series.value_counts(ascending = False)

director_all    = df['director'].str.cat(sep = '|')
director_series = pd.Series(director_all.split('|'))
director_count  = director_series.value_counts(ascending = False)

genre_all    = df['genres'].str.cat(sep = '|')
genre_series = pd.Series(genre_all.split('|'))
genre_count  = genre_series.value_counts(ascending = False)

keyword_all    = df['keywords'].str.cat(sep = '|')
keyword_series = pd.Series(keyword_all.split('|'))
keyword_count  = keyword_series.value_counts(ascending = False)

production_co_all    = df['production_companies'].str.cat(sep = '|')
production_co_series = pd.Series(production_co_all.split('|'))
production_co_count  = production_co_series.value_counts(ascending = False)

plt.figure()
actor_count[0:10].plot(kind='bar')
plt.title('Top 10 Movie Actors')
plt.xlabel('Actor')
plt.ylabel('Movie Count')
plt.grid(True)

plt.figure()
director_count[0:10].plot(kind='bar')
plt.title('Top 10 Movie Directors')
plt.xlabel('Director')
plt.ylabel('Movie Count')
plt.grid(True)

plt.figure()
genre_count[0:10].plot(kind='bar')
plt.title('Top 10 Movie Genre')
plt.xlabel('Genre')
plt.ylabel('Movie Count')
plt.grid(True)

plt.figure()
keyword_count[0:10].plot(kind='bar')
plt.title('Top 10 TMDb Keywords')
plt.xlabel('TMDb Keyword')
plt.ylabel('Movie Count')
plt.grid(True)

plt.figure()
production_co_count[0:10].plot(kind='bar')
plt.title('Top 10 Movie Production Companies')
plt.xlabel('Production Company')
plt.ylabel('Movie Count')
plt.grid(True)


# Q. Which genres are most popular from year to year?

# ordered list of genres
genre = genre_count.index.to_list()
# list of all genres per movie
movie_genres = list(map(str, (df['genres'])))
# year column per movie
year_vector = np.array(df['release_year'])
# IMDb pop per movie
pop_vector  = np.array(df['popularity'])

# empty DF = genre X year(s)
pop_year_df = pd.DataFrame(0.0, index=genre, columns=range(1960, 2016))
cou_year_df = pd.DataFrame(0.0, index=genre, columns=range(1960, 2016))
# fill DF columns with movie popularity
j = 0
for i in movie_genres:
    # for every movie, split genre string into a LIST of multiple movies
    movie_genres_split = list(map(str,i.split('|')))
    # populate DF by adding the IMDb popularity (or a 1.0 count) to that particular genre and year
    pop_year_df.loc[movie_genres_split, year_vector[j]] = pop_year_df.loc[movie_genres_split, year_vector[j]] + pop_vector[j]
    cou_year_df.loc[movie_genres_split, year_vector[j]] = cou_year_df.loc[movie_genres_split, year_vector[j]] + 1.0
    j = j + 1

fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15,6))
fig.suptitle('IMDb Popularity Index per Year')
k = 0
for i in range(3):
    for j in range(5):
        pop_year_df.loc[genre[k]].plot(label=genre[k], color='red', ax=ax[i][j], legend=True)
        ax[i][j].set_ylim(0, 200)
        ax[i][j].legend(fontsize=10, loc='upper left')
        k = k + 1

fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15,6))
fig.suptitle('Number of Movie Releases per Year')
k = 0
for i in range(3):
    for j in range(5):
        cou_year_df.loc[genre[k]].plot(label=genre[k], color='red', ax=ax[i][j], legend=True)
        ax[i][j].set_ylim(0, 100)
        ax[i][j].legend(fontsize=10, loc='upper left')
        k = k + 1