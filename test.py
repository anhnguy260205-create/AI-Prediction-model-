
import yfinance as yf

ticker = yf.Ticker("META")
hist = ticker.history(period="1mo")

print(hist.empty)
print(hist.tail())
