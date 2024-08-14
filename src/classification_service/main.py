from fastapi import FastAPI, status

from ticket_model import HeathCheck, Ticket, TicketClassification, TicketModel

app = FastAPI()
ticket_model = TicketModel()


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Heath Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HeathCheck,
)
async def get_health() -> HeathCheck:
    """## Perform a Health Check

    Returns:
    --------
        HeathCheck: Returns a JSON response with the health status
    """
    return HeathCheck(status="OK")


@app.post("/ticket")
async def create_item(ticket: Ticket):
    """## Perform a Ticket prediction

    Returns:
    --------
        TicketClassification: Returns a JSON response with the prediction and probability
    """
    prediction, probability = ticket_model.predict(ticket.description)
    return TicketClassification(prediction=prediction, probability=probability)
