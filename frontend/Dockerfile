# Multi-stage build for React frontend with Vite and Bun

# Stage 1: Build stage
FROM oven/bun:latest AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json bun.lockb ./

# Install dependencies
RUN bun install

# Copy source code
COPY . .

# Build the application
RUN bun run build

# Stage 2: Production stage with nginx
FROM nginx:alpine AS production

# Copy custom nginx configuration
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config (optional - you can create this if needed)
# COPY nginx.conf /etc/nginx/nginx.conf

# Create a simple nginx configuration for SPA routing
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Alternative development stage (uncomment if you need development mode)
# FROM oven/bun:latest AS development
# WORKDIR /app
# COPY package.json bun.lockb ./
# RUN bun install
# COPY . .
# EXPOSE 5173
# CMD ["bun", "run", "dev", "--host"]