use crossbeam::channel;
use crypto_trading_bot::{self, client::Client, exchange::SimulatedExchange};

fn main() {
    let (to_client, from_exchange) = channel::unbounded();
    let (to_exchange, from_client) = channel::unbounded();

    let exchange = SimulatedExchange::new(to_client, from_client);
    let client = Client::new(to_exchange, from_exchange);

    crossbeam::scope(|scope| {
        scope.spawn(|_| {
            exchange.run();
        });

        scope.spawn(|_| {
            client.run();
        });

        std::thread::sleep(std::time::Duration::from_secs(60));
        std::process::exit(0);
    })
    .unwrap();
}
