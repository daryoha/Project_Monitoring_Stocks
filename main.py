from header import *
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

  @bot.poll_answer_handler(func=lambda call: True)
  async def handle_poll_answer(pollAnswer):
      ops=pollAnswer.option_ids
      res=[]
      msg = types.Message(message_id=0, from_user=pollAnswer.user.id, date='', chat = pollAnswer.user.id, content_type='text', options=[], json_string='')
      if choice[pollAnswer.user.id]=="Котировки":
          for x in ops:
              res.append(comp_list[x])
          #
          new_r = pickle.dumps(res)
          conn = sqlite3.connect('it_user.sql')
          cur = conn.cursor() # ссылка на контекстную область памяти #работае синх асинх
          cur.execute('UPDATE users SET companies_cot = ? WHERE id = ?', (new_r, pollAnswer.user.id))
          conn.commit() #синхроним
          cur.close()
          conn.close()
          #
          msg.text='Компании для котировок'
      elif choice[pollAnswer.user.id]=="Уведомления":
          for x in ops:
              res.append(comp_dict[comp_list[x]])
          #
          new_r = pickle.dumps(res)
          conn = sqlite3.connect('it_user.sql')
          cur = conn.cursor() # ссылка на контекстную область памяти #работае синх асинх
          cur.execute('UPDATE users SET companies_info = ? WHERE id = ?', (new_r, pollAnswer.user.id))
          conn.commit() #синхроним
          cur.close()
          conn.close()
          #
          msg.text='Компании для уведомлений'
      await get_messages(msg)

  @bot.message_handler(content_types=['text', "test"])
  async def get_messages(message):
      if message.text == "Отлично! Начнем работу :)":
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Посмотреть котировки')
          btn2 = types.KeyboardButton('Построить прогноз')
          btn3 = types.KeyboardButton('Настроить уведомления')
          markup.add(btn1, btn2, btn3)
          await bot.send_message(message.from_user.id, 'Выберите, что вы хотите сделать', reply_markup=markup)

      elif message.text == 'Посмотреть котировки':
          choice[message.chat.id]="Котировки"
          await bot.send_poll(message.chat.id, question="Выберите, акции каких компаний вы хотите мониторить",
              options=comp_list, allows_multiple_answers=True, is_anonymous=False)

      elif message.text=="Настроить уведомления":
          choice[message.chat.id]="Уведомления"
          await bot.send_poll(message.chat.id, question="Выберите, новости о каких компаниях вы хотите получать",
              options=comp_list, allows_multiple_answers=True, is_anonymous=False)

      if message.text == "Компании для котировок":
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
          btn1 = types.KeyboardButton('Последний день')
          btn2 = types.KeyboardButton('Последняя неделя')
          btn3 = types.KeyboardButton('Последний месяц')
          markup.add(btn1, btn2, btn3)
          await bot.send_message(message.from_user, 'Выберите временной промежуток', reply_markup=markup)

      elif message.text.split()[0][:7]=="Последн":
          #
          user_id = message.from_user.id
          conn = sqlite3.connect('it_user.sql')
          cur = conn.cursor() # ссылка на контекстную область памяти #работае синх асинх
          cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
          table = cur.fetchall()

          user_cot = table[0][1]
          cur.close()
          conn.close()
          user_cot = pickle.loads(user_cot)
          #Высчитываем временной промежуток
          end_date = date.today().isoformat()
          start_date=0
          time_range=message.text.split()[1]

          if (time_range=="день"):
               start_date = (date.today()-timedelta(days=1)).isoformat()
          elif (time_range=="неделя"):
               start_date = (date.today()-timedelta(days=7)).isoformat()
          elif (time_range=="месяц"):
               start_date = (date.today()-timedelta(days=30)).isoformat()
          data_user=data_days[data_days["Date"]>=start_date]
          quote_comp_user = user_cot
          for comp_name in quote_comp_user:
              comp=data_user[["Date", comp_dict[comp_name]]]
              text=(f'Компания: {comp_name}\n'
                    f'Период: {time_range}\n')

              plt.figure(figsize=(6, 4))
              line=sns.lineplot(data=comp, x='Date', y=comp_dict[comp_name])
              line.set(xlabel='Date', ylabel='Stock price')
              line.set_title(f'Stock price dynamics for {comp_name} company', fontdict={'size': 10, 'weight': 'bold', 'color': 'green', 'style':'italic'})
              plt.xticks(rotation=90)
              filename=f'{comp_name}.png'
              plt.savefig(filename, bbox_inches='tight')
              PATH=filename
              im=pyimgur.Imgur(CLIENT_ID)
              uploaded_image=im.upload_image(PATH, title=PATH)
              await bot.send_message(user_id, text)
              await bot.send_photo(user_id, uploaded_image.link)
              os.remove(filename)
      elif message.text=="Компании для уведомлений":
          #
          user_id = message.from_user
          conn = sqlite3.connect('it_user.sql')
          cur = conn.cursor() # ссылка на контекстную область памяти #работае синх асинх
          cur.execute('SELECT * FROM users WHERE id = ?', (user_id))
          table = cur.fetchall()
          user_info = table[0][2]
          cur.close()
          conn.close()
          user_info = pickle.loads(user_info)
          #
          notif_comp_user=user_info
  await bot.polling(none_stop=True, interval=0)

asyncio.run(main())
