# Use a lightweight Nginx image
FROM nginx:alpine

# Create a directory for website content
WORKDIR /usr/share/nginx/html

# Copy all website files from the current directory
# COPY . .

# Expose port 80 where your website will be served
EXPOSE 80

# Run Nginx in the foreground
CMD [ "nginx", "-g", "daemon off;" ]