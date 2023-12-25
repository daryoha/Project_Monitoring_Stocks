from header import *
import header
#ДАШЕ: функция, которая рисует график с предсказаниями
def prediction_func(company):
    prediction_result=predict_fn(company, comp_dict, quotes_day)
    fig = go.Figure(go.Scatter(x=prediction_result['Date'], y=prediction_result[comp_dict[company]],
                        name=f"Предсказанная динамика акций компании {company} на 7 дней"))
    filename=f"{company}.png"
    fig.write_image(filename)
    PATH=filename
    im=pyimgur.Imgur(CLIENT_ID)
    uploaded_image=im.upload_image(PATH, title=PATH)
    return uploaded_image
async def main():
  @bot.message_handler(commands=['start']) #
  async def start(message):
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
      #
      user_id = message.from_user.id
      conn = sqlite3.connect('it_user.sql')
      cur = conn.cursor() # ссылка на контекстную область памяти #работае синх асинх
      cur.execute(f'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, companies_cot BLOB, companies_info BLOB);')
      cur.execute(f'SELECT * FROM users WHERE id = ?', (user_id,));
      tables = cur.fetchall()
      if not tables:
          cur.execute(f"INSERT INTO users (id,companies_cot,companies_info) VALUES ({user_id} ,NULL,NULL)")
      conn.commit() #синхроним
      cur.close()
      conn.close()
      #
      btn1 = types.KeyboardButton("Отлично! Начнем работу :)")
      markup.add(btn1)
      await bot.send_message(message.from_user.id, "Здравствуйте! Этот бот поможет вам отслеживать котировки акций.", reply_markup=markup)

  @bot.message_handler(content_types=['text', "test"])
  async def get_messages(message):
      if message.text == "Отлично! Начнем работу :)":
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Посмотреть котировки')
          btn2 = types.KeyboardButton('Построить прогноз')
          markup.add(btn1, btn2)
          await bot.send_message(message.from_user.id, 'Выберите, что вы хотите сделать', reply_markup=markup)

      elif message.text == 'Посмотреть котировки':
          choice[message.chat.id]="Котировки"
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          buttons = list()
          for name in comp_dict.values():
              buttons.append(types.KeyboardButton(name))
          markup.add(*buttons)
          await bot.send_message(message.from_user.id, 'Выберите, что вы хотите сделать', reply_markup=markup)

      elif message.text=="Построить прогноз":
          choice[message.chat.id]="Предсказания"
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Яндекс')
          btn2 = types.KeyboardButton('Газпром')
          btn3 = types.KeyboardButton('Лукойл')
          btn4 = types.KeyboardButton('Сбербанк')
          btn5 = types.KeyboardButton('Русал')
          btn6 = types.KeyboardButton('Интер РАО')
          btn7 = types.KeyboardButton('Норильский никель')
          btn8 = types.KeyboardButton('Магнит')
          btn9 = types.KeyboardButton('Полюс')
          btn10 = types.KeyboardButton('МТС')
          btn11 = types.KeyboardButton('Северсталь')
          btn12 = types.KeyboardButton('НОВАТЭК')
          btn13 = types.KeyboardButton('Татнефть')
          btn14 = types.KeyboardButton('Сургетнефтегаз')
          markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, btn10, btn11, btn12, btn13, btn14)
          await bot.send_message(message.from_user.id, 'Выберите, что вы хотите сделать', reply_markup=markup)

      if message.text in comp_list_pred:
          user_id = message.from_user.id
          text = (f'Компания: {message.text}\n')
          uploaded_image = prediction_func(message.text)
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Посмотреть котировки')
          btn2 = types.KeyboardButton('Построить прогноз')
          markup.add(btn1, btn2)
          await bot.send_message(user_id, text)
          await bot.send_photo(user_id, uploaded_image.link, reply_markup=markup)

      if message.text in comp_dict.values():
          header.company = message.text
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Последняя неделя')
          btn2 = types.KeyboardButton('Последний месяц')
          markup.add(btn1, btn2)
          await bot.send_message(message.from_user.id, 'Выберите временной промежуток', reply_markup=markup)

      if message.text == "Компании для котировок":
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Последняя неделя')
          btn2 = types.KeyboardButton('Последний месяц')
          markup.add(btn1, btn2)
          await bot.send_message(message.from_user, 'Выберите временной промежуток', reply_markup=markup)

      elif message.text.split()[0][:7]=="Последн":
          user_id = message.from_user.id
          end_date = date.today().isoformat()
          start_date=0
          time_range=message.text.split()[1]

          if (time_range=="неделя"):
               start_date = (date.today()-timedelta(days=7)).isoformat()
          elif (time_range=="месяц"):
               start_date = (date.today()-timedelta(days=30)).isoformat()
          data_user=data_days[data_days["Date"]>=start_date]
          comp=data_user[["Date", header.company]]
          text=(f'Компания: {header.company}\n'
                f'Период: {time_range}\n')
          plt.figure(figsize=(6, 4))
          line=sns.lineplot(data=comp, x='Date', y=header.company)
          line.set(xlabel='Date', ylabel='Stock price')
          line.set_title(f'Stock price dynamics for {header.company} company', fontdict={'size': 10, 'weight': 'bold', 'color': 'green', 'style':'italic'})
          plt.xticks(rotation=90)
          filename=f'{header.company}.png'
          plt.savefig(filename, bbox_inches='tight')
          PATH=filename
          im=pyimgur.Imgur(CLIENT_ID)
          uploaded_image=im.upload_image(PATH, title=PATH)
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Посмотреть котировки')
          btn2 = types.KeyboardButton('Построить прогноз')
          markup.add(btn1, btn2)
          await bot.send_message(user_id, text)
          await bot.send_photo(user_id, uploaded_image.link, reply_markup=markup)
          os.remove(filename)

  await bot.polling(none_stop=True, interval=0)

asyncio.run(main())

