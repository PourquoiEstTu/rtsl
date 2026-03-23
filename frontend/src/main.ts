import { createApp } from "vue";
import "./style.css";
import App from "./App.vue";
import PrimeVue from "primevue/config";
import "primeicons/primeicons.css";
import router from "./router";
import ToastService from "primevue/toastservice";

const app = createApp(App);
app.use(PrimeVue, {
  unstyled: true,
});
app.use(router);
app.use(ToastService);
app.mount("#app");
