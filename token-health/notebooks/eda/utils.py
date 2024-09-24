import matplotlib.pyplot as plt
import pandas as pd

def plot_token_data(df, token_name, cutoff_date):
    """Plot TVL, Market Cap, and Price for a specific token.

    Args:
        df (pd.DataFrame): DataFrame containing token data.
        token_name (str): The name of the token to filter.
        cutoff_date (pd.Timestamp): The date for the vertical line.
    """
    # Filter by token
    df_tok = df[df['Name'] == token_name]
    
    # Create a figure with a single axis and secondary axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Set labels and tick parameters for TVL axis
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('TVL', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second axis for Market Cap
    ax2 = ax1.twinx()
    ax2.set_ylabel('Market Cap', color='violet')
    ax2.plot(df_tok['Date'], df_tok['Market cap'], color='violet', label='Market Cap')
    ax2.tick_params(axis='y', labelcolor='violet')
    
    # Create a third axis for Price
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Price', color='indigo')
    ax3.plot(df_tok['Date'], df_tok['Price'], color='indigo', label='Price')
    ax3.tick_params(axis='y', labelcolor='indigo')
    
    # Plot the TVL line with green or red color based on the correlation
    for i in range(1, len(df_tok)):
        color = 'green' if df_tok['Correlation'].iloc[i] > 0 else 'black'
        ax1.plot(df_tok['Date'].iloc[i-1:i+1], df_tok['TVL'].iloc[i-1:i+1], color=color, label='TVL' if i == 1 else "")
    
    # Add vertical line on the specified cutoff date
    ax1.axvline(x=cutoff_date, linestyle='--', color='red', label=cutoff_date.strftime('%d %B %Y'))
    
    # Title and caption
    fig.suptitle(f'Evolution of TVL, Market Cap, and Price for {token_name}')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper center')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()