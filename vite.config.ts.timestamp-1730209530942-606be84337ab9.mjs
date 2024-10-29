// vite.config.ts
import { defineConfig, loadEnv } from "file:///workspaces/JARVIS/node_modules/vite/dist/node/index.js";
import topLevelAwait from "file:///workspaces/JARVIS/node_modules/vite-plugin-top-level-await/exports/import.mjs";
var vite_config_default = defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  const proxyConf = env.VITE_QUEUE_API_URL ? {
    "/api": {
      target: env.VITE_QUEUE_API_URL,
      changeOrigin: true
    }
  } : {};
  return {
    server: {
      host: "0.0.0.0",
      https: {
        cert: "./cert.pem",
        key: "./key.pem"
      },
      proxy: {
        ...proxyConf
      }
    },
    plugins: [
      topLevelAwait({
        // The export name of top-level await promise for each chunk module
        promiseExportName: "__tla",
        // The function to generate import names of top-level await promise in each chunk module
        promiseImportName: (i) => `__tla_${i}`
      })
    ]
  };
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvd29ya3NwYWNlcy9KQVJWSVNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZmlsZW5hbWUgPSBcIi93b3Jrc3BhY2VzL0pBUlZJUy92aXRlLmNvbmZpZy50c1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vd29ya3NwYWNlcy9KQVJWSVMvdml0ZS5jb25maWcudHNcIjtpbXBvcnQgeyBQcm94eU9wdGlvbnMsIGRlZmluZUNvbmZpZywgbG9hZEVudiB9IGZyb20gXCJ2aXRlXCI7XG5pbXBvcnQgdG9wTGV2ZWxBd2FpdCBmcm9tIFwidml0ZS1wbHVnaW4tdG9wLWxldmVsLWF3YWl0XCI7XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZygoe21vZGV9KSA9PiB7XG4gIGNvbnN0IGVudiA9IGxvYWRFbnYobW9kZSwgcHJvY2Vzcy5jd2QoKSk7XG4gIGNvbnN0IHByb3h5Q29uZjpSZWNvcmQ8c3RyaW5nLCBzdHJpbmcgfCBQcm94eU9wdGlvbnM+ID0gZW52LlZJVEVfUVVFVUVfQVBJX1VSTCA/IHtcbiAgICBcIi9hcGlcIjoge1xuICAgICAgdGFyZ2V0OiBlbnYuVklURV9RVUVVRV9BUElfVVJMLFxuICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxuICAgIH0sXG4gIH0gOiB7fTtcbiAgcmV0dXJuIHtcbiAgICBzZXJ2ZXI6IHtcbiAgICAgIGhvc3Q6IFwiMC4wLjAuMFwiLFxuICAgICAgaHR0cHM6IHtcbiAgICAgICAgY2VydDogXCIuL2NlcnQucGVtXCIsXG4gICAgICAgIGtleTogXCIuL2tleS5wZW1cIixcbiAgICAgIH0sXG4gICAgICBwcm94eTp7XG4gICAgICAgIC4uLnByb3h5Q29uZixcbiAgICAgIH1cbiAgICB9LFxuICAgIHBsdWdpbnM6IFtcbiAgICAgIHRvcExldmVsQXdhaXQoe1xuICAgICAgICAvLyBUaGUgZXhwb3J0IG5hbWUgb2YgdG9wLWxldmVsIGF3YWl0IHByb21pc2UgZm9yIGVhY2ggY2h1bmsgbW9kdWxlXG4gICAgICAgIHByb21pc2VFeHBvcnROYW1lOiBcIl9fdGxhXCIsXG4gICAgICAgIC8vIFRoZSBmdW5jdGlvbiB0byBnZW5lcmF0ZSBpbXBvcnQgbmFtZXMgb2YgdG9wLWxldmVsIGF3YWl0IHByb21pc2UgaW4gZWFjaCBjaHVuayBtb2R1bGVcbiAgICAgICAgcHJvbWlzZUltcG9ydE5hbWU6IGkgPT4gYF9fdGxhXyR7aX1gLFxuICAgICAgfSksXG4gICAgXSxcbiAgfTtcbn0pO1xuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUF3TyxTQUF1QixjQUFjLGVBQWU7QUFDNVIsT0FBTyxtQkFBbUI7QUFFMUIsSUFBTyxzQkFBUSxhQUFhLENBQUMsRUFBQyxLQUFJLE1BQU07QUFDdEMsUUFBTSxNQUFNLFFBQVEsTUFBTSxRQUFRLElBQUksQ0FBQztBQUN2QyxRQUFNLFlBQWtELElBQUkscUJBQXFCO0FBQUEsSUFDL0UsUUFBUTtBQUFBLE1BQ04sUUFBUSxJQUFJO0FBQUEsTUFDWixjQUFjO0FBQUEsSUFDaEI7QUFBQSxFQUNGLElBQUksQ0FBQztBQUNMLFNBQU87QUFBQSxJQUNMLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLE9BQU87QUFBQSxRQUNMLE1BQU07QUFBQSxRQUNOLEtBQUs7QUFBQSxNQUNQO0FBQUEsTUFDQSxPQUFNO0FBQUEsUUFDSixHQUFHO0FBQUEsTUFDTDtBQUFBLElBQ0Y7QUFBQSxJQUNBLFNBQVM7QUFBQSxNQUNQLGNBQWM7QUFBQTtBQUFBLFFBRVosbUJBQW1CO0FBQUE7QUFBQSxRQUVuQixtQkFBbUIsT0FBSyxTQUFTLENBQUM7QUFBQSxNQUNwQyxDQUFDO0FBQUEsSUFDSDtBQUFBLEVBQ0Y7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
