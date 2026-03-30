import { UserController } from './controller/UserController.js';
import { MovieController } from './controller/MovieController.js';
import { ModelController } from './controller/ModelTrainingController.js';
import { TFVisorController } from './controller/TFVisorController.js';
import { TFVisorView } from './view/TFVisorView.js';
import { UserService } from './service/UserService.js';
import { MovieService } from './service/MovieService.js';
import { UserView } from './view/UserView.js';
import { MovieView } from './view/MovieView.js';
import { ModelView } from './view/ModelTrainingView.js';
import Events from './events/events.js';
import { WorkerController } from './controller/WorkerController.js';

// Create shared services
const userService = new UserService();
const movieService = new MovieService();

// Create views
const userView = new UserView();
const movieView = new MovieView();
const modelView = new ModelView();
const tfVisorView = new TFVisorView();
const mlWorker = new Worker('/src/workers/modelTrainingWorker.js', { type: 'module' });

// Set up worker message handler
const w = WorkerController.init({
    worker: mlWorker,
    events: Events
});

const users = await userService.getDefaultUsers();
w.triggerTrain(users);


ModelController.init({
    modelView,
    userService,
    events: Events,
});

TFVisorController.init({
    tfVisorView,
    events: Events,
});

MovieController.init({
    movieView,
    userService,
    movieService,
    events: Events,
});


const userController = UserController.init({
    userView,
    userService,
    productService: movieService,
    events: Events,
});


userController.renderUsers({
    "id": 99,
    "name": "Josézin da Silva",
    "age": 30,
    "movieWatches": []
});