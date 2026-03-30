import { View } from './View.js';

export class UserView extends View {
    #userSelect = document.querySelector('#userSelect');
    #userAge = document.querySelector('#userAge');
    #pastPurchasesList = document.querySelector('#pastPurchasesList');

    #purchaseTemplate;
    #onUserSelect;
    #onPurchaseRemove;
    #pastPurchaseElements = [];

    constructor() {
        super();
        this.init();
    }

    async init() {
        this.#purchaseTemplate = await this.loadTemplate('./src/view/templates/past-purchase.html');
        this.attachUserSelectListener();
    }

    registerUserSelectCallback(callback) {
        this.#onUserSelect = callback;
    }

    registerPurchaseRemoveCallback(callback) {
        this.#onPurchaseRemove = callback;
    }

    renderUserOptions(users) {
        debugger
        const options = users.map(user => {
            return `<option value="${user.id}">${user.name}</option>`;
        }).join('');

        this.#userSelect.innerHTML += options;
    }

    renderUserDetails(user) {
        this.#userAge.value = user.age;
    }

    renderPastPurchases(pastPurchases) {
        if (!this.#purchaseTemplate) return;

        if (!pastPurchases || pastPurchases.length === 0) {
            this.#pastPurchasesList.innerHTML = '<p>No Watched Movies found.</p>';
            return;
        }

        const html = pastPurchases.map(movie => {
            return this.replaceTemplate(this.#purchaseTemplate, {
                ...movie,
                movie: JSON.stringify(movie)
            });
        }).join('');

        this.#pastPurchasesList.innerHTML = html;
        this.attachPurchaseClickHandlers();
    }

    addPastPurchase(movie) {

        if (this.#pastPurchasesList.innerHTML.includes('No Watched Movies found')) {
            this.#pastPurchasesList.innerHTML = '';
        }

        const purchaseHtml = this.replaceTemplate(this.#purchaseTemplate, {
            ...movie,
            movie: JSON.stringify(movie)
        });

        this.#pastPurchasesList.insertAdjacentHTML('afterbegin', purchaseHtml);

        const newPurchase = this.#pastPurchasesList.firstElementChild.querySelector('.past-purchase');
        newPurchase.classList.add('past-purchase-highlight');

        setTimeout(() => {
            newPurchase.classList.remove('past-purchase-highlight');
        }, 1000);

        this.attachPurchaseClickHandlers();
    }

    attachUserSelectListener() {
        this.#userSelect.addEventListener('change', (event) => {
            const userId = event.target.value ? Number(event.target.value) : null;

            if (userId) {
                if (this.#onUserSelect) {
                    this.#onUserSelect(userId);
                }
            } else {
                this.#userAge.value = '';
                this.#pastPurchasesList.innerHTML = '';
            }
        });
    }

    attachPurchaseClickHandlers() {
        this.#pastPurchaseElements = [];

        const purchaseElements = document.querySelectorAll('.past-purchase');

        purchaseElements.forEach(purchaseElement => {
            this.#pastPurchaseElements.push(purchaseElement);

            purchaseElement.onclick = (event) => {

                const movie = JSON.parse(purchaseElement.dataset.movie);
                const userId = this.getSelectedUserId();
                const element = purchaseElement.closest('.col-md-6');

                this.#onPurchaseRemove({ element, userId, movie });

                element.style.transition = 'opacity 0.5s ease';
                element.style.opacity = '0';

                setTimeout(() => {
                    element.remove();

                    if (document.querySelectorAll('.past-purchase').length === 0) {
                        this.renderPastPurchases([]);
                    }

                }, 500);

            }
        });
    }

    getSelectedUserId() {
        return this.#userSelect.value ? Number(this.#userSelect.value) : null;
    }
}
