from __future__ import annotations

import re
import traceback
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command
from aiogram.enums import ParseMode

from .config import load_config
from .forecasting.data import load_prices
from .forecasting.selector import select_best_model
from .forecasting.strategy import buy_sell_points, simulate_profit
from .forecasting.plotting import plot_history_and_forecast
from .utils.logging import append_log_row, utc_now_iso

HELP_TEXT = (
    "Отправьте сообщение в формате:\n"
    "<тикер компании> <сумму денег для условной инвестиции>\n\n"
    "Примеры:\n"
    "AAPL 1000\n"
    "MSFT 25000\n"
)

INPUT_RE = re.compile(r"^\s*([A-Za-z0-9\-\.]{1,15})\s+([0-9]+(?:\.[0-9]+)?)\s*$")


async def cmd_start(message: Message):
    await message.answer(
        "Привет. Я бот для прогнозирования цен акций на основе временных рядов.\n\n" + HELP_TEXT
    )


async def cmd_help(message: Message):
    await message.answer(HELP_TEXT)


async def handle_text(message: Message, bot: Bot, log_path: str):
    m = INPUT_RE.match(message.text or "")
    if not m:
        await message.answer("Некорректные данные. Попробуйте еще раз.\n\n" + HELP_TEXT)
        return

    ticker = m.group(1).upper()
    amount = float(m.group(2))
    user_id = message.from_user.id if message.from_user else "unknown"

    try:
        await message.answer(
            f"Обработка тикера {ticker} с суммой {amount:.2f}. Загрузка данных и обучение моделей..."
        )

        series = load_prices(ticker, years=2)

        best_model, results = select_best_model(series)

        forecast = best_model.forecast(series, steps=30)

        buy_dates, sell_dates = buy_sell_points(forecast, order=1)
        profit = simulate_profit(forecast, amount, buy_dates, sell_dates)

        change_pct = (forecast.iloc[-1] - series.iloc[-1]) / series.iloc[-1] * 100.0

        buf = plot_history_and_forecast(series, forecast, title=f"{ticker}: история + прогноз")

        await bot.send_photo(
            chat_id=message.chat.id,
            photo=BufferedInputFile(buf.getvalue(), filename=f"{ticker}_forecast.png"),
            caption=f"{ticker}: Прогноз готов. Ожидаемое изменение по сравнению с ценой закрытия предыдущего дня: {change_pct:+.2f}%",
        )

        lines = []
        lines.append(f"<b>{ticker}</b>")
        lines.append(f"Последняя цена: <b>{series.iloc[-1]:.2f}</b>")
        lines.append(f"Прогноз (последние 30 дней): <b>{forecast.iloc[-1]:.2f}</b> ({change_pct:+.2f}%)")
        lines.append("")
        lines.append("<b>Сравнение моделей (сортировка по RMSE):</b>")
        for r in results:
            det = f" ({r.details})" if r.details else ""
            lines.append(f"- {r.name}{det}: RMSE={r.rmse:.4f}, MAPE={r.mape:.2f}%")
        lines.append("")
        if buy_dates and sell_dates:
            lines.append("<b>Рекомендации:</b>")
            for b, s in zip(buy_dates, sell_dates):
                lines.append(
                    f"- Покупать: {b.date()} @ {forecast.loc[b]:.2f}  → Подавать: {s.date()} @ {forecast.loc[s]:.2f}"
                )
        else:
            lines.append(
                "<b>Рекомендации:</b> В 30-дневном прогнозе недостаточно четко выраженных локальных минимумов/максимумов.")
        lines.append("")
        lines.append(f"<b>Ожидаемая прибыль</b>: {profit:+.2f} на сумму {amount:.2f}")

        await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)

        append_log_row(
            log_path,
            {
                "timestamp_utc": utc_now_iso(),
                "user_id": str(user_id),
                "ticker": ticker,
                "amount": f"{amount:.2f}",
                "best_model": getattr(best_model, "name", best_model.__class__.__name__),
                "metric_primary": "RMSE",
                "rmse": f"{results[0].rmse:.6f}",
                "mape": f"{results[0].mape:.6f}",
                "profit_estimate": f"{profit:.2f}",
                "buy_dates": ",".join([d.strftime("%Y-%m-%d") for d in buy_dates]),
                "sell_dates": ",".join([d.strftime("%Y-%m-%d") for d in sell_dates]),
            },
        )

    except Exception as e:
        await message.answer(
            "Произошла ошибка при обработке запроса. Пожалуйста, проверьте тикер и повторите попытку.\n\n"
            f"Подробности: {type(e).__name__}: {e}"
        )
        append_log_row(
            log_path,
            {
                "timestamp_utc": utc_now_iso(),
                "user_id": str(user_id),
                "ticker": ticker,
                "amount": f"{amount:.2f}",
                "best_model": "",
                "metric_primary": "",
                "rmse": "",
                "mape": "",
                "profit_estimate": "",
                "buy_dates": "",
                "sell_dates": "",
            },
        )


def main():
    cfg = load_config()
    bot = Bot(token=cfg.telegram_token)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))

    async def on_text(message: Message, bot: Bot):
        await handle_text(message, bot, cfg.log_path)

    dp.message.register(on_text, F.text)

    dp.run_polling(bot)


if __name__ == "__main__":
    main()
