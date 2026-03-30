import { View } from './View.js';

export class MovieView extends View {
    // DOM elements
    #movieList = document.querySelector('#productList');

    #buttons;
    // Templates and callbacks
    #movieTemplate;
    #onBuyProduct;

    constructor() {
        super();
        this.init();
    }

    async init() {
        this.#movieTemplate = await this.loadTemplate('./src/view/templates/product-card.html');
    }

    onUserSelected(user) {
        // Enable buttons if a user is selected, otherwise disable them
        this.setButtonsState(user.id ? false : true);
    }

    registerBuyProductCallback(callback) {
        this.#onBuyProduct = callback;
    }

    render(movies, disableButtons = true) {
        if (!this.#movieTemplate) return;
        const html = movies.map(movie => {
            return this.replaceTemplate(this.#movieTemplate, {
                id: movie.id,
                name: movie.name,
                category: movie.category,
                price: movie.price,
                color: movie.color,
                movie: JSON.stringify(movie)
            });
        }).join('');

        this.#movieList.innerHTML = html;
        this.attachBuyButtonListeners();

        // Disable all buttons by default
        this.setButtonsState(disableButtons);
    }

    setButtonsState(disabled) {
        if (!this.#buttons) {
            this.#buttons = document.querySelectorAll('.buy-now-btn');
        }
        this.#buttons.forEach(button => {
            button.disabled = disabled;
        });
    }

    attachBuyButtonListeners() {
        this.#buttons = document.querySelectorAll('.buy-now-btn');
        this.#buttons.forEach(button => {

            button.addEventListener('click', (event) => {
                const product = JSON.parse(button.dataset.product);
                const originalText = button.innerHTML;

                button.innerHTML = '<i class="bi bi-check-circle-fill"></i> Added';
                button.classList.remove('btn-primary');
                button.classList.add('btn-success');
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-primary');
                }, 500);
                this.#onBuyProduct(product, button);

            });
        });
    }
}
