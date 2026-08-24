import { type Customer } from "../domain/customer.ts";
import { CustomerHttpClient } from "../infrastructure/customerHttpClient.ts";

export class CustomerService {
    private readonly client: CustomerHttpClient;

    constructor(baseUrl: string) {
        this.client = new CustomerHttpClient(baseUrl);
    }

    async listCustomers(): Promise<Customer[]> {
        return this.client.listCustomers()
    }
}