import argparse
import logging
from game.engine import PokerEngine
from ai.agent import PokerAgent
from config import Config
import matplotlib.pyplot as plt
import sys
import traceback
import os


def configure_logging():
    if Config.DETAILED_REPORTING:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_training_progress(evaluation_results):
    episodes, rewards, win_rates, elo_ratings = zip(*evaluation_results)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

    ax1.plot(episodes, rewards)
    ax1.set_title('Average Reward')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')

    ax2.plot(episodes, win_rates)
    ax2.set_title('Win Rate')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Win Rate')

    ax3.plot(episodes, elo_ratings)
    ax3.set_title('ELO Rating')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('ELO')

    # Additional metrics
    avg_rewards = [reward for _, reward, _, _ in evaluation_results]
    avg_win_rates = [win_rate for _, _, win_rate, _ in evaluation_results]
    avg_elo_ratings = [elo for _, _, _, elo in evaluation_results]

    ax4.plot(episodes, avg_rewards, label='Avg Reward')
    ax4.plot(episodes, avg_win_rates, label='Avg Win Rate')
    ax4.plot(episodes, avg_elo_ratings, label='Avg ELO')
    ax4.set_title('Average Metrics')
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Metrics')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="Poker AI")
    parser.add_argument("--train", action="store_true", help="Train the AI")
    parser.add_argument("--play", action="store_true", help="Play against the AI")
    args = parser.parse_args()

    config = Config()
    if not config.DETAILED_REPORTING and not config.VERBOSE:
        config.optimize_for_speed()
    engine = PokerEngine(config)
    logging.info("PokerEngine initialized")

    agent = PokerAgent(config, engine)
    logging.info("PokerAgent initialized")

    try:
        if args.train:
            logging.info("Starting AI training...")
            evaluation_results = agent.train()
            if evaluation_results:
                plot_training_progress(evaluation_results)
                logging.info("Training completed. Progress plot saved as 'training_progress.png'")
            else:
                logging.warning("No evaluation results were produced during training.")
        elif args.play:
            logging.info("Starting game against AI...")
            engine.play_vs_human(agent)
        else:
            logging.warning("Please specify --train or --play")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving current model...")
        agent.save("interrupted_model.pth")
        logging.info("Model saved as 'interrupted_model.pth'. Exiting...")
        sys.exit(0)
    except ValueError as ve:
        logging.error(f"ValueError occurred: {str(ve)}")
    except TypeError as te:
        logging.error(f"TypeError occurred: {str(te)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Exception details:")
        traceback.print_exc()
    finally:
        logging.info("Main process has been terminated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[!] {os.path.basename(sys.argv[0])}: process terminated")
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Exception details:")
        traceback.print_exc()
