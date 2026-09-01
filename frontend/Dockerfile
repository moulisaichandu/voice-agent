# The operator console. Built and run as a compose service so `docker compose
# up` brings up the whole system — before this, frontend/ was a complete Next
# app that nobody ran, and every campaign was created by hand with curl.
FROM node:22-slim AS deps
WORKDIR /console
# Copied alone so a dependency layer is reused whenever only app code changed.
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-slim AS build
WORKDIR /console
COPY --from=deps /console/node_modules ./node_modules
COPY . .
# NODE_ENV=production makes next.config.js write to .next-build rather than
# .next, so a build here can never clobber a `next dev` running on the host
# against the same bind-mounted directory.
ENV NODE_ENV=production
RUN npm run build

FROM node:22-slim AS run
WORKDIR /console
ENV NODE_ENV=production
COPY --from=deps /console/node_modules ./node_modules
COPY --from=build /console/.next-build ./.next-build
COPY package.json next.config.js ./
EXPOSE 3200
CMD ["npm", "run", "start"]
