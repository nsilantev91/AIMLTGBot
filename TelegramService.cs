using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using System.Drawing;
using Newtonsoft.Json;
namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }
        public MagicEye processor = new MagicEye();
        BaseNetwork Net;

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;
            // Net = new StudentNetwork(@"../../NN.data");
            var s = System.IO.File.ReadAllText(@"../../NN.txt");
            Net = Newtonsoft.Json.JsonConvert.DeserializeObject<StudentNetwork>(s);

        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, messageText),
                    cancellationToken: cancellationToken);
                return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                // Если бы мы хотели получить битмап, то могли бы использовать new Bitmap(Image.FromStream(imageStream))
                // Но вместо этого пошлём картинку назад
                // Стрим помнит последнее место записи, мы же хотим теперь прочитать с самого начала
                var orig = new Bitmap(imageStream);
                
                processor.ProcessImage(orig);
                var processed = processor.processed;
                processed.Save(@"../../file.png");
                var sample = CreateSample(processed);
                Net.Predict(sample);
                await client.SendTextMessageAsync(chatId: message.Chat.Id, sample.recognizedClass.ToString());

                //set_result(sample);
                /*imageStream.Seek(0, 0);
                await client.SendPhotoAsync(
                    message.Chat.Id,
                    imageStream,
                    "Пока что я не знаю, что делать с картинками, так что держи обратно",
                    cancellationToken: cancellationToken
                );*/
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        public Sample CreateSample(Bitmap image, FigureType actualType = FigureType.Undef)
        {
            var inputs = new double[200];
            var img = new Bitmap(image);
            for (int i = 0; i < 100; i++)
            {
                inputs[i] = CountBlackPixels(GetBitmapColumn(img, i));
                inputs[i + 100] = CountBlackPixels(GetBitmapRow(img, i));
            }

            return new Sample(inputs, 5, actualType);
        }

        public int CountBlackPixels(Color[] pixels) =>
          pixels.Count(p => p.R < 0.1 && p.G < 0.1 && p.B < 0.1);

        public Color[] GetBitmapColumn(Bitmap picture, int ind)
        {
            var result = new Color[picture.Height];
            for (int i = 0; i < picture.Height; i++)
                result[i] = picture.GetPixel(ind, i);
            return result;
        }

        public Color[] GetBitmapRow(Bitmap picture, int ind)
        {
            var result = new Color[picture.Width];
            for (int i = 0; i < picture.Width; i++)
                result[i] = picture.GetPixel(i, ind);
            return result;
        }
        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }
    }
}
