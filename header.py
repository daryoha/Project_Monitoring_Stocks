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
from telebot.async_telebot import AsyncTeleBot
from datetime import date, timedelta
data_days = pd.read_csv('fin_quotes_per_day.csv')
CLIENT_ID="f45d25c17f9968c"
TOKEN = '6896623536:AAH4QCmavIiBgdkg8v5CVOECX14cM1JjpG4'
bot = AsyncTeleBot(TOKEN)
comp_list=["Yandex", "Газпром", "Лукойл", "Сбербанк", "Русал", "Интер РАО", "Норильский никель", "Магнит", "МТС"]
comp_dict={"Yandex":"YNDX", "Газпром":"GAZP", "Лукойл":"LKOH", "Сбербанк":"SBER", "Русал":"RUAL", "Интер РАО":"IRAO", "Норильский никель":"GMKN", "Магнит":"MGNT", "МТС":"MTSS"}
notif_comp={}
choice={}