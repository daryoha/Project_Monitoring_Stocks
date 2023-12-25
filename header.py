from dl_models.model_main import train_all, predict_fn
import pandas as pd
import numpy as np
import telebot
from telebot import types
from dateutil.parser import parse
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random
import requests, json
import pyimgur
import os
import sqlite3 #
import pickle
import aiohttp
import asyncio
import plotly.graph_objects as go
from telebot.async_telebot import AsyncTeleBot
from datetime import date, timedelta
data_days = pd.read_csv('fin_quotes_per_day.csv')
CLIENT_ID="f45d25c17f9968c"
TOKEN = '6759361009:AAH1NZP6Xjk92bVCLEKUcpOUrFhw8WhYBFA'
bot = AsyncTeleBot(TOKEN)
comp_list=["Yandex", "Газпром", "Лукойл", "Сбербанк", "Русал", "Интер РАО", "Норильский никель", "Магнит", "МТС"]
comp_dict={"Yandex":"YNDX", "Газпром":"GAZP", "Лукойл":"LKOH", "Сбербанк":"SBER", "Русал":"RUAL", "Интер РАО":"IRAO",
           "Норильский никель":"GMKN", "Магнит":"MGNT", "МТС":"MTSS", "Татнефть":'TATN', "Сургутнефтегаз":'SNGS',
           "НОВАТЭК":'NVTK', "Полюс":'PLZL', "Северсталь":'CHMF'}
comp_list_pred=["Яндекс", "Газпром", "Лукойл", "Сбербанк", "Русал", "Интер РАО", "Норильский никель",
                "Магнит", "МТС", "Полюс", "Роснефть", "Северсталь", "Сургутнефтегаз", "НОВАТЭК", "Татнефть"]
quotes_day=pd.read_csv("fin_quotes_per_day.csv")
notif_comp={}
choice={}
company=""