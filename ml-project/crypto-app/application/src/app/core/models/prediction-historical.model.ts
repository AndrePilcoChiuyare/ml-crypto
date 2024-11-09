export interface PredictionHistorical {
    id: string;
    category: string;
    future_multiply: number;
    has_increased: number;
    last_close: number;
    last_pred_close: number;
    market_cap_level: string;
    marketcap: number;
    name: string;
    symbol: string;
    close_data: {
        close:number;
        timestamp: string;
    }[];
}