import z from "zod";

export const CustomerSchema = z.object({
    _id: z.string().optional(),
    name: z.string(),
    phone: z.string(),
})

export type Customer = z.infer<typeof CustomerSchema>