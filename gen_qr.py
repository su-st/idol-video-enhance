import qrcode
url = "https://atlantic-overarch-unwind.ngrok-free.dev"
img = qrcode.make(url)
img.save("qrcode_ngrok.png")
print("QR Code saved: qrcode_ngrok.png")
