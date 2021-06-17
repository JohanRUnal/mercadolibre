<p align="center">
  <img width="320" height="160" src="https://play-lh.googleusercontent.com/vJw7auvtuNBhjdgLBw_V_pfWHqAAkZiZ0ftWuUF_ZTiJUOT0COnJa0iZCHoC_BtSFH4">
</p>

Mercadolibre is absolutely one of the most important companies of e-commerce in the world. For many years it has been a reference of innovation and still look itself as a "*Startup*", where there are still a lot of work to do, curiously that simple but powerful point of view definitely is a captstone of its philosophy and success!.

In this project we are interested in analyse its great public API called **MELI** , where we can access to some information of a huge amount of selling posts that are already allow in their current platform. Our goal will be create a "*full pipeline*" as is described below.

- **Data collection**: Collect data from the MELI API using a scraper.

- **Preprocessing and data cleaning** : Clean the raw data and make some feature engineering tasks. 
- **Exploratory analysis**: We are interested in the extraction of insights, patterns and anomalies.
- **Modelling stage**: We'll try to create a model that  let us "*predict*" the **sold quantity** of products that belong to different categories.

### Tools and libraries.
<img src="https://img.shields.io/badge/-Python-brightgreen"> | <img src="https://img.shields.io/badge/-sklear-green"> | <img src="https://img.shields.io/badge/-pandas-yellow"> | <img src="https://img.shields.io/badge/-seaborn-red"> | <img src="https://img.shields.io/badge/-matplotlib-purple">

### Some conclusions:
1. In general Adaboost showed a better performance in contrast with traditional regression models to predict the sold quantity variable getting a mae(mean absoulute error) so close to zero.
2. Data was taken from the mercadolibre API, so we unknown the "diversity" of the data extracted so that some overfit problems definitely can take place.


### Future works:

Try to validate with the API owner the diversity of the data and find other data sources useful to get robust models.



