# Frontend Folder

This folder contains all of the code used to develop our front end and the integration with the backend. There are two primary folders, the "public" folder consists of images and models, and the "src" folder consists of all the code that models the front end.

Within the "src" folder, the "components" folder consists of some of the key components used in modeling the front end. This folder has a respective tests subfolder as well. The "screens" folder consists of the main screens the user will see, which includes the Home Page, Translator page, and About Us. This folder also has a subfolder called "style" for CSS files for each screen, and a "tests" subfolder for unit tests.

All code related to the web extension version of RTSL can be found in the "extension" folder within "src".

If running the front end locally, run "npm run dev" in the /frontend folder. The same applies for "npm run test"

## What this folder is for

- Build and run the front end
- Run unit tests on the front end
- Model the logic for sending of data and receiving words / sentences
