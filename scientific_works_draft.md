
# **1\. Introduction**

## **1.1. Problem area**

Recent developments of machine learning algorithms bring new approaches for analysing financial data including algorithmic trading, which is becoming one of the most applicable fields for data scientists. The combination of plenty of different types of data (from time series to news) at financial markets and speed competition of implemented models make this field ambitious yet extra-difficult. In algorithmic trading, models seek to identify and exploit micro-level signals in real time. However, the problem lies in the sheer volume and complexity of streaming data. Market events arrive on the order of microseconds, making it difficult to extract actionable patterns using traditional econometric models, therefore it is needed to use advanced machine learning techniques.

Recent scientific works show that deep learning improves forecasting of market dynamics but requires additional sources, such as news aggregation. For instance, CNNs and LSTMs have been applied to financial data and social media data with success but just for hourly and daily frequency (Ortu et al., 2022)\[1\]. Analysing social media indicators for smaller time intervals becomes irrational since it requires large computational power, therefore for algorithmic trading another additional data source should be used. One possible extension is emphasizing the relational structure of market data: financial systems can be viewed as graphs where nodes represent assets or stocks, and edges represent correlations or relations between companies (Wang et al., 2020)\[2\]. Graph neural networks (GNNs) therefore provide a natural framework for capturing these interactions. But usually GNNs are considered as static graphs \[3\]-\[4\]-\[5\], which limits them to use for stationary environments only. 

Temporal Graph Neural Networks (TGNNs) extend this approach by incorporating event-driven updates and memory mechanisms, enabling efficient adaptation to streaming data (Huang et al., 2020)\[6\]. Some scientists have already applied TGNNs to non-financial forecasting \[7\]-\[8\] with success . More recent scientific papers also successfully apply this approach to financial structures (Qian et al., 2024)\[9\], but for daily frequency. 

Although TGNNs have been shown to work well in dynamic graph tasks, their application to real-time market microstructure data remains underexplored. The vast of studies use offline data or introduce TGNNs for rarely updating (24-hours) intervals instead of seconds. **Therefore, there is a significant gap in scientific works that consider TGNNs with 1 second \- 1 minute frequency updating.**

To examine this problem memory mechanisms, and multigraph fusion based on TGNNs for streaming data is suggested. In practice these extensions can be integrated into TGNN architectures specifically for real-time financial microstructure modeling.

The proposed thesis will address the research gap of building a frequently-updating TGNN based on streaming financial data for market dynamics prediction.

## **2.2. Data description**

[Kaggle](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) **\- High-Frequency Crypto Limit Order Book Data**

* **Content:** High-frequency order book snapshots for multiple cryptocurrency pairs (BTC/USD, ETH/USD, etc.) collected from major exchanges.  
* **Structure:** Includes top-10 bid/ask levels, order volumes, and trades, stored as CSV files   
* **Granularity:** second-level updates, capturing the rapid dynamics of crypto markets.  
* **Usefulness**: Provides a different asset class (cryptocurrencies), which are known for higher volatility and 24/7 trading. Useful for testing the generalization ability of microstructure models beyond equities.

# Literature  

\[1\] Ortu, M., Uras, N., Conversano, C., Bartolucci, S. and Destefanis, G., 2022\. On technical trading and social media indicators for cryptocurrency price classification through deep learning. *Expert Systems with Applications*, *198*, p.116804.  
\[2\] Wang, J., Zhang, S., Xiao, Y. and Song, R., 2021\. A review on graph neural network methods in financial applications. arXiv preprint arXiv:2111.15367.  
\[3\] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X. and Zhang, C., 2020, August. Connecting the dots: Multivariate time series forecasting with graph neural networks. In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 753-763). [Link](https://arxiv.org/pdf/2005.11650)  
\[4\] Khemani, B., Patil, S., Kotecha, K. and Tanwar, S., 2024\. A review of graph neural networks: concepts, architectures, techniques, challenges, datasets, applications, and future directions. *Journal of Big Data*, *11*(1), p.18. [Link](https://link.springer.com/content/pdf/10.1186/s40537-023-00876-4.pdf)  
\[5\] Zhang, Z., Cui, P. and Zhu, W., 2020\. Deep learning on graphs: A survey. IEEE Transactions on Knowledge and Data Engineering, 34(1), pp.249-270. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9039675&casa_token=_R-fBk_myCcAAAAA:zvluRDBNSm9Qpnm7UebIoFdimNNPWFHWulCjcyr0Pkfw6ppyjgNGB18ZPyBwta-xQogYanVLkB97CA&tag=1)  
\[6\] Huang, R., Rossi, E., Fey, M., Hamilton, W. and Bronstein, M. (2020) Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637. [Link](https://arxiv.org/pdf/2006.10637)  
\[7\] Liu, Y., Liu, Q., Zhang, J.W., Feng, H., Wang, Z., Zhou, Z. and Chen, W., 2022\. Multivariate time-series forecasting with temporal polynomial graph neural networks. *Advances in neural information processing systems*, *35*, pp.19414-19426. [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b102c908e9404dd040599c65db4ce3e-Paper-Conference.pdf)  
\[8\] Jin, M., Zheng, Y., Li, Y.F., Chen, S., Yang, B. and Pan, S., 2022\. Multivariate time series forecasting with dynamic graph neural odes. *IEEE Transactions on Knowledge and Data Engineering*, *35*(9), pp.9168-9180. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9950330&casa_token=93jLwK0vfR4AAAAA:sZZ9UP_GzXY4LmgKh4rS6CtlhluGJUqwZoeoXod4qG05akXedP1mFK2NB5oGbbcgYwYjOERsWaUtLA&tag=1)  
\[9\] Qian, H., Zhou, H., Zhao, Q., Chen, H., Yao, H., Wang, J., Liu, Z., Yu, F., Zhang, Z. and Zhou, J., 2024, March. Mdgnn: Multi-relational dynamic graph neural network for comprehensive and dynamic stock investment prediction. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 13, pp. 14642-14650). [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29381)

