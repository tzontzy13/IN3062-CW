import os

import numpy as np
import pandas as pd

#read file
path = "."
filename_read = os.path.join(path, "covid_cleaned.csv")
df = pd.read_csv(filename_read)

#reshuflle
df = df.reindex(np.random.permutation(df.index))

# drop columns
df.drop('contact_other_covid', 1, inplace=True)
df.drop('Unnamed: 0', 1, inplace=True)

# filter data, we only keep rows with a "positive" covid_res
covid = df[df['covid_res']==1.0]

# since all covid_res have value 1, we drop column
covid.drop('covid_res', 1, inplace=True)

# sort
sorted_df = covid.sort_values(by='age')

print(sorted_df[0:5])
print(sorted_df[-5:])