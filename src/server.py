from peak import Agent, CyclicBehaviour, JoinGroup
from settings import group

class server(Agent):
    class InitFederation(CyclicBehaviour):
        async def on_start(self) -> None:
            await self.wait_for(JoinGroup())
            while not len(self.agent.group_members()) >= self.n_agents:
                await _asyncio.sleep(1)
            self.current_period = 0
            _logger.info("Starting simulation...")
        async def run(self) -> None:
            pass
