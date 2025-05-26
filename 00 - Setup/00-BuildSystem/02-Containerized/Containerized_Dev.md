The other option for development is containerized environments. 
This is a more advanced setup that allows you to run the development environment in a container, which can be useful for isolating dependencies and ensuring consistency across different machines.
## Containerized Development Environment
### Prerequisites
- Docker installed on your machine
- Docker Compose installed on your machine
### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate into the project directory:
   ```bash
   cd <project-directory>
   ```
3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
4. Start the development environment:
   ```bash
   docker-compose up -d
   ```
5. Access the container:
   ```bash
    docker exec -it <container-name> /bin/bash
    ```
6. Inside the container, you can run your development commands, such as building the project or running tests.
7. To stop the development environment, run:
   ```bash
   docker-compose down
   ```

## Development Workflow
- Host Side:
  - Use your preferred code editor to edit files in the project directory.
  - The changes will be reflected inside the container due to the volume mount specified in the `docker-compose.yml` file.
- Container Side:
  - Use the terminal inside the container to run build commands, tests, or any other development tasks.
  - You can install additional dependencies as needed using the package manager inside the container.

### Notes
- Ensure that your Docker daemon is running before starting the container.
- You can customize the `.env` file to change environment variables as needed.
- The containerized environment is designed to be used for development purposes and may not be suitable for production use.
- If you need to install additional dependencies, you can modify the Dockerfile or use the package manager inside the container.
- For more advanced configurations, you can modify the `docker-compose.yml` file to suit your needs.
- This setup allows you to work in a consistent environment regardless of your host machine's configuration.
- If you encounter issues with file permissions, you may need to adjust the user and group settings in the Dockerfile or use volume mounts with appropriate permissions.
- You can also use Docker volumes to persist data across container restarts.    
