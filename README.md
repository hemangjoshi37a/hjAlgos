# hjAlgos

![image](https://github.com/user-attachments/assets/93f8c0b1-9de8-40d6-9b9a-cf2f094bbf7b)


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/hjAlgos.svg?style=social&label=Star)](https://github.com/yourusername/hjAlgos)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/hjAlgos.svg?style=social&label=Fork)](https://github.com/yourusername/hjAlgos/fork)

**hjAlgos** is an open-source algorithmic trading platform that leverages advanced machine learning models to predict stock prices and execute trades in real-time. Integrated with Zerodha for trading and Appwrite for backend services, hjAlgos provides a seamless and automated trading experience for both novice and experienced traders.

## 🚀 Features

- **Advanced Stock Prediction:** Utilizes a Transformer-based neural network for accurate stock price predictions.
- **Real-Time Trading:** Automatically executes buy and sell orders based on the latest predictions.
- **User-Friendly Interface:** Intuitive web interface built with Flask and Bokeh for visualizing predictions and trade history.
- **Backtesting:** Analyze historical performance of trading strategies with comprehensive backtest results.
- **Secure and Scalable:** Employs environment variables for secure configuration management and scalable backend services.

## 📈 Demo and Backtests

- **Live Demo:** [https://hjalgos.hjlabs.in](https://hjalgos.hjlabs.in)
- **Backtest Results:** [https://hjalgos.hjlabs.in/backtest/](https://hjalgos.hjlabs.in/backtest/)
![image](https://github.com/user-attachments/assets/9daa4d2f-ccb0-4b44-89bc-59b83e5d4690)
![image](https://github.com/user-attachments/assets/c886124c-01fe-43da-88ff-b7e82d5a40e8)



## 🛠 Installation

### Prerequisites

- **Python 3.8+**
- **Zerodha Account** with API access
- **Appwrite Account**
- **GPU** (optional, for faster model inference)

### Clone the Repository

```bash
git clone https://github.com/yourusername/hjAlgos.git
cd hjAlgos
```

### Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup Environment Variables

Create a `.env` file in the root directory based on the provided `.env.example`.

```bash
cp .env.example .env
```

Edit the `.env` file and fill in your credentials:

```dotenv
# .env

# Flask Secret Key
SECRET_KEY=your_very_secure_secret_key

# Appwrite Configuration
APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1
APPWRITE_PROJECT_ID=your_project_id
APPWRITE_API_KEY=your_api_key
APPWRITE_DATABASE_ID=your_database_id

# Zerodha Credentials
ZERODHA_USER_ID=your_zerodha_user_id
ZERODHA_PASSWORD=your_zerodha_password
ZERODHA_TOTP_KEY=your_totp_key
```

> **⚠️ Security Tip:**  
> **Never commit your `.env` file to version control systems like GitHub.** To prevent accidental exposure, ensure `.env` is listed in your `.gitignore` file.

### Initialize Appwrite Collections

Ensure that the required collections (`users`, `trades`, `predictions`) are created in your Appwrite project.

### Load Normalization Parameters

Ensure that `mean.pkl` and `std.pkl` files are present in the project directory. These files are used for normalizing input data.

### Load the Pre-trained Model

Ensure that `stock_predictor_model.pth` is present in the project directory.

## 📂 Project Structure

```
hjAlgos/
├── app.py
├── predictor.py
├── templates/
│   └── index.html
├── static/
│   └── images/
│       └── totp_info.png
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## 💡 Usage

### Running the Predictor

The `predictor.py` script continuously fetches stock data, runs predictions using the ML model, and saves predictions to the Appwrite database.

```bash
python predictor.py
```

### Running the Web Application

The `app.py` script runs the Flask web application, providing a user interface for managing trading sessions, viewing predictions, and monitoring trade history.

```bash
python app.py
```

Access the web app at [http://localhost:5000](http://localhost:5000)

## 📝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create your feature branch:**

    ```bash
    git checkout -b feature/YourFeature
    ```

3. **Commit your changes:**

    ```bash
    git commit -m 'Add some feature'
    ```

4. **Push to the branch:**

    ```bash
    git push origin feature/YourFeature
    ```

5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## 🧾 License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

## 📫 How to Reach Me

[<img height="36" src="https://cdn.simpleicons.org/similarweb"/>](https://hjlabs.in/) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/WhatsApp"/>](https://wa.me/917016525813) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/telegram"/>](https://t.me/hjlabs) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Gmail"/>](mailto:hemangjoshi37a@gmail.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/LinkedIn"/>](https://www.linkedin.com/in/hemang-joshi-046746aa) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/facebook"/>](https://www.facebook.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Twitter"/>](https://twitter.com/HemangJ81509525) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/tumblr"/>](https://www.tumblr.com/blog/hemangjoshi37a-blog) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/StackOverflow"/>](https://stackoverflow.com/users/8090050/hemang-joshi) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Instagram"/>](https://www.instagram.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Pinterest"/>](https://in.pinterest.com/hemangjoshi37a) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Blogger"/>](http://hemangjoshi.blogspot.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/gitlab"/>](https://gitlab.com/hemangjoshi37a) &nbsp;

## 🤝 Sponsorship

This project is sponsored by [hjLabs](https://hjlabs.in).

---

