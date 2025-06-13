from mcp.server.fastmcp import FastMCP


# Initialize the FastMCP server with the service name "Weather"
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Retrieve weather information for the specified location.

    Args:
        location (str): The location for which to retrieve the weather.

    Returns:
        str: A string describing the weather.
    """
    return "It's always sunny in New York"


if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run(transport="stdio")
