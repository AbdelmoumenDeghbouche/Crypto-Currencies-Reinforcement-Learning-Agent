# README.md

## Crypto Currencies Reinforcement Learning Agent

This repository demonstrates how to create and train a reinforcement learning (RL) agent to trade cryptocurrency using historical data. The project leverages advanced RL techniques, including a Long Short-Term Memory (LSTM) policy with Proximal Policy Optimization (PPO). It also features a custom-built trading environment created entirely from scratch and compatible with the OpenAI Gym API.

### Features
- **Custom Trading Environment**: Implements a fully customizable trading simulation environment.
- **Advanced RL Algorithm**: Uses the LSTM-based PPO algorithm from the Stable-Baselines3 Contrib library.
- **Yahoo Finance Integration**: Fetches and preprocesses cryptocurrency data dynamically.
- **Custom Indicators**: Includes SMA, RSI, and OBV as features for the agent.
- **Detailed Evaluation**: Tracks and visualizes rewards, profit percentages, and final net worth over time.

---

## Dependencies
Install the required libraries:
```bash
pip install stable-baselines3 gym gymnasium finta yfinance
```

---

## Project Overview

### 1. Data Collection and Preprocessing
- **Source**: Yahoo Finance
- **Data Features**:
  - Open, High, Low, Close, Volume
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - On-Balance Volume (OBV)

#### Example Code:
```python
btc_data = yf.download('BTC-USD', start='2024-01-01', end='2025-01-15')
btc_data['SMA'] = TA.SMA(btc_data, 12)
btc_data['RSI'] = TA.RSI(btc_data)
btc_data['OBV'] = TA.OBV(btc_data)
btc_data.fillna(0, inplace=True)
```

---

### 2. Custom Trading Environment
The environment is built entirely from scratch using the Gym API. Key components include:
- **Actions**: Buy, Sell
- **Positions**: Short, Long
- **Reward Calculation**: Based on price differences and position changes.
- **Profit Tracking**: Tracks cumulative profits and calculates gain percentage.

#### Environment Features:
- Render functions to visualize trades.
- Customizable starting balance and trade fees.

#### Environment Code Example:
```python
class TradingEnv(gym.Env):
    def __init__(self, df, window_size, render_mode=None):
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.action_space = gym.spaces.Discrete(2)  # Buy, Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.signal_features.shape[1]), dtype=np.float32
        )
        self.render_mode = render_mode
        self._reset()

    def step(self, action):
        # Implement trading logic, reward calculation, and profit tracking
        pass

    def reset(self):
        # Reset environment state
        pass

    def _process_data(self):
        # Preprocess input data (prices and indicators)
        pass

    def render(self, mode='human'):
        # Visualize trades and profits
        pass
```

---

### 3. Training the Agent
- **Algorithm**: Recurrent PPO
- **Policy**: MlpLstmPolicy
- **Training Parameters**:
  - Learning Rate: `2e-4`
  - Number of Steps: `4096`
  - Batch Size: `32`
  - Entropy Coefficient: `0.02`

#### Example Code:
```python
ppo_params = {
    "policy": "MlpLstmPolicy",
    "env": vec_env,
    "learning_rate": 2e-4,
    "n_steps": 4096,
    "batch_size": 32,
    "ent_coef": 0.02,
    "verbose": 1
}
ppo_model = RecurrentPPO(**ppo_params)
ppo_model.learn(total_timesteps=100000, callback=eval_callback)
ppo_model.save("ppo_lstm_crypto_model")
```

---

### 4. Evaluation
The trained model is tested in a new environment using unseen data. Key metrics such as cumulative reward, profit percentage, and final net worth are visualized.

#### Example Code:
```python
def testing_env_and_model(model):
    test_btc_data = preprocess_data()
    test_env = TradingEnv(df=test_btc_data, window_size=5)
    obs, info = test_env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            break
```

---

### 5. Visualization
Final results are visualized to display:
- **Rewards over time**
- **Profit percentage**
- **Final net worth**

#### Example Plot:
Below is the final plot showing rewards and profit trends over time:
```python
plt.figure(figsize=(15,6))
plt.cla()
test_env.render_all()
plt.show()
```

![Final Results Plot](Plots/final_results_plot.png)

---

## Repository Structure
```
|-- data/                     # Directory for storing raw and processed data
|-- models/                   # Trained models
|-- scripts/                  # Scripts for training and evaluation
|-- notebooks/                # Jupyter notebooks for experimentation
|-- README.md                 # Project documentation (this file)
|-- requirements.txt          # Python dependencies
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or script to train the model.

---

## Acknowledgments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Finta Library](https://github.com/peerchemist/finta)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

