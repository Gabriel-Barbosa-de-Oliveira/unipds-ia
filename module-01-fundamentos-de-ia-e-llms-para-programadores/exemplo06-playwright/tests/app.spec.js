const { test, expect } = require('@playwright/test');

// baseURL is https://erickwendel.github.io/vanilla-js-web-app-example/
// './' resolves to that full path; '/' would resolve to the GitHub Pages root and return 404
const APP_URL = './';

test.beforeEach(async ({ page }) => {
  await page.goto(APP_URL);
  // clear localStorage so previous test runs don't affect card counts
  await page.evaluate(() => localStorage.removeItem('tdd-ew-db'));
  await page.reload();
});

test('loads the correct page', async ({ page }) => {
  await expect(page).toHaveURL(/vanilla-js-web-app-example/);
  await expect(page).toHaveTitle('TDD Frontend Example');
});

test('shows the form inputs and submit button', async ({ page }) => {
  await expect(page.getByRole('textbox', { name: 'Image Title' })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Image URL' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Submit Form' })).toBeVisible();
});

test('displays the three default image cards', async ({ page }) => {
  const cardTitles = page.locator('.card-title');
  await expect(cardTitles).toHaveCount(3);
  await expect(cardTitles.nth(0)).toHaveText('AI Alien');
  await expect(cardTitles.nth(1)).toHaveText('Predator Night Vision');
  await expect(cardTitles.nth(2)).toHaveText('ET Bilu');
});

test('adds a new card when the form is submitted', async ({ page }) => {
  const titleInput = page.getByRole('textbox', { name: 'Image Title' });
  const urlInput = page.getByRole('textbox', { name: 'Image URL' });
  const submitButton = page.getByRole('button', { name: 'Submit Form' });

  await titleInput.fill('My Test Image');
  await urlInput.fill('https://picsum.photos/300');
  await submitButton.click();

  const cardTitles = page.locator('.card-title');
  await expect(cardTitles).toHaveCount(4);
  await expect(page.getByRole('heading', { level: 4, name: 'My Test Image' })).toBeVisible();
});

test('does not submit when form fields are empty', async ({ page }) => {
  const submitButton = page.getByRole('button', { name: 'Submit Form' });
  await submitButton.click();

  // form stays on the same page and no new card is added
  await expect(page).toHaveURL(/vanilla-js-web-app-example/);
  await expect(page.locator('.card-title')).toHaveCount(3);
});

test('prevents submission when image URL is invalid', async ({ page }) => {
  const titleInput = page.getByRole('textbox', { name: 'Image Title' });
  const urlInput = page.getByRole('textbox', { name: 'Image URL' });
  const submitButton = page.getByRole('button', { name: 'Submit Form' });

  await titleInput.fill('Invalid URL Image');
  await urlInput.fill('not-a-url');
  await submitButton.click();

  await expect(page.locator('.card-title')).toHaveCount(3);
  await expect(page.getByRole('heading', { level: 4, name: 'Invalid URL Image' })).toHaveCount(0);
});
