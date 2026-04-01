"""
Windows Task Scheduler Automation
Creates, manages, and removes scheduled tasks for MT5 data collection
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from typing import Optional
import tempfile
import logging

# Get the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class TaskSchedulerManager:
    """
    Manages Windows Task Scheduler tasks for MT5 data collection.

    Uses Windows schtasks.exe command-line utility.
    """

    TASK_NAME = "MT5_DataCollector"

    def __init__(
        self,
        task_name: str = None,
        python_path: str = None,
        script_path: str = None,
        working_dir: str = None
    ):
        self.task_name = task_name or self.TASK_NAME
        self.python_path = python_path or sys.executable
        self.script_path = script_path or os.path.join(PROJECT_DIR, 'mt5_collector.py')
        self.working_dir = working_dir or PROJECT_DIR

    def _run_schtasks(self, args: list, check: bool = True) -> subprocess.CompletedProcess:
        """Run schtasks command with given arguments"""
        cmd = ['schtasks'] + args
        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"schtasks error: {e.stderr}")
            raise

    def task_exists(self) -> bool:
        """Check if the scheduled task already exists"""
        result = self._run_schtasks(
            ['/Query', '/TN', self.task_name],
            check=False
        )
        return result.returncode == 0

    def create_hourly_task(
        self,
        interval_hours: int = 1,
        start_time: str = "00:00"
    ) -> bool:
        """
        Create a task that runs every N hours.

        Args:
            interval_hours: Hours between runs (1-23)
            start_time: Start time in HH:MM format

        Returns:
            True if successful
        """
        if self.task_exists():
            logger.warning(f"Task '{self.task_name}' already exists. Deleting...")
            self.delete_task()

        # Create batch script wrapper
        batch_content = self._create_batch_script()
        batch_path = os.path.join(self.working_dir, 'run_collector.bat')

        with open(batch_path, 'w') as f:
            f.write(batch_content)

        logger.info(f"Created batch script: {batch_path}")

        # Create scheduled task
        args = [
            '/Create',
            '/TN', self.task_name,
            '/TR', f'"{batch_path}"',
            '/SC', 'HOURLY',
            '/MO', str(interval_hours),
            '/ST', start_time,
            '/F'  # Force create (overwrite if exists)
        ]

        try:
            result = self._run_schtasks(args)
            logger.info(f"Created hourly task '{self.task_name}' (every {interval_hours}h)")
            return True
        except subprocess.CalledProcessError:
            return False

    def create_daily_task(
        self,
        times: list = None,
        days: str = "MON,TUE,WED,THU,FRI"
    ) -> bool:
        """
        Create task(s) that run at specific times each day.

        Args:
            times: List of times in HH:MM format (e.g., ["08:00", "12:00", "18:00"])
            days: Days to run (MON,TUE,WED,THU,FRI,SAT,SUN)

        Returns:
            True if successful
        """
        if times is None:
            times = ["06:00", "12:00", "18:00", "23:00"]

        # Create batch script
        batch_content = self._create_batch_script()
        batch_path = os.path.join(self.working_dir, 'run_collector.bat')

        with open(batch_path, 'w') as f:
            f.write(batch_content)

        success = True

        for i, time in enumerate(times):
            task_name = f"{self.task_name}_{i+1}"

            # Delete if exists
            if self._task_exists_by_name(task_name):
                self._delete_task_by_name(task_name)

            args = [
                '/Create',
                '/TN', task_name,
                '/TR', f'"{batch_path}"',
                '/SC', 'WEEKLY',
                '/D', days,
                '/ST', time,
                '/F'
            ]

            try:
                self._run_schtasks(args)
                logger.info(f"Created task '{task_name}' for {time} on {days}")
            except subprocess.CalledProcessError:
                success = False
                logger.error(f"Failed to create task for {time}")

        return success

    def create_interval_task(
        self,
        interval_minutes: int = 30
    ) -> bool:
        """
        Create a task that runs every N minutes.

        Args:
            interval_minutes: Minutes between runs (1-1439)

        Returns:
            True if successful
        """
        if self.task_exists():
            logger.warning(f"Task '{self.task_name}' already exists. Deleting...")
            self.delete_task()

        batch_path = os.path.join(self.working_dir, 'run_collector.bat')
        batch_content = self._create_batch_script()

        with open(batch_path, 'w') as f:
            f.write(batch_content)

        args = [
            '/Create',
            '/TN', self.task_name,
            '/TR', f'"{batch_path}"',
            '/SC', 'MINUTE',
            '/MO', str(interval_minutes),
            '/F'
        ]

        try:
            self._run_schtasks(args)
            logger.info(f"Created task '{self.task_name}' (every {interval_minutes} minutes)")
            return True
        except subprocess.CalledProcessError:
            return False

    def delete_task(self) -> bool:
        """Delete the scheduled task"""
        return self._delete_task_by_name(self.task_name)

    def delete_all_tasks(self) -> int:
        """Delete all MT5 collector tasks. Returns number deleted."""
        deleted = 0

        # Delete main task
        if self.task_exists():
            self.delete_task()
            deleted += 1

        # Delete numbered tasks (for daily schedules)
        for i in range(1, 10):
            task_name = f"{self.task_name}_{i}"
            if self._task_exists_by_name(task_name):
                self._delete_task_by_name(task_name)
                deleted += 1

        return deleted

    def _task_exists_by_name(self, name: str) -> bool:
        """Check if a specific task exists"""
        result = self._run_schtasks(['/Query', '/TN', name], check=False)
        return result.returncode == 0

    def _delete_task_by_name(self, name: str) -> bool:
        """Delete a specific task by name"""
        try:
            self._run_schtasks(['/Delete', '/TN', name, '/F'])
            logger.info(f"Deleted task '{name}'")
            return True
        except subprocess.CalledProcessError:
            return False

    def run_task_now(self) -> bool:
        """Manually trigger the task to run immediately"""
        try:
            self._run_schtasks(['/Run', '/TN', self.task_name])
            logger.info(f"Task '{self.task_name}' triggered")
            return True
        except subprocess.CalledProcessError:
            return False

    def get_task_status(self) -> Optional[dict]:
        """Get current status of the scheduled task"""
        if not self.task_exists():
            return None

        result = self._run_schtasks(
            ['/Query', '/TN', self.task_name, '/V', '/FO', 'LIST'],
            check=False
        )

        if result.returncode != 0:
            return None

        # Parse the output
        status = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                status[key.strip()] = value.strip()

        return status

    def _create_batch_script(self) -> str:
        """Create the batch script content for running the collector"""
        return f'''@echo off
REM MT5 Data Collector - Scheduled Task Runner
REM Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

cd /d "{self.working_dir}"

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
)

REM Run the collector
"{self.python_path}" "{self.script_path}" --skip-db-setup

REM Log completion
echo [%date% %time%] Collection completed >> "{self.working_dir}\\logs\\scheduler.log"
'''

    def create_xml_task(
        self,
        schedule_type: str = "hourly",
        interval: int = 1,
        start_time: str = "00:00"
    ) -> str:
        """
        Create an XML task definition for more complex scheduling.
        Returns the path to the XML file.
        """
        # Generate XML content
        xml_content = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>MT5 OHLCV Data Collector - Automated data collection</Description>
    <Author>MT5 Data Collector</Author>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT{interval}H</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{datetime.now().strftime("%Y-%m-%d")}T{start_time}:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{self.working_dir}\\run_collector.bat"</Command>
      <WorkingDirectory>{self.working_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
'''

        # Save XML file
        xml_path = os.path.join(self.working_dir, 'task_schedule.xml')
        with open(xml_path, 'w', encoding='utf-16') as f:
            f.write(xml_content)

        logger.info(f"Created XML task definition: {xml_path}")
        return xml_path

    def import_xml_task(self, xml_path: str) -> bool:
        """Import a task from XML file"""
        try:
            self._run_schtasks([
                '/Create',
                '/TN', self.task_name,
                '/XML', xml_path,
                '/F'
            ])
            logger.info(f"Imported task from {xml_path}")
            return True
        except subprocess.CalledProcessError:
            return False


def print_status(manager: TaskSchedulerManager):
    """Print current task status"""
    print("\n" + "=" * 60)
    print("MT5 Data Collector - Task Scheduler Status")
    print("=" * 60)

    status = manager.get_task_status()

    if status:
        print(f"Task Name: {manager.task_name}")
        print(f"Status: {status.get('Status', 'Unknown')}")
        print(f"Last Run: {status.get('Last Run Time', 'Never')}")
        print(f"Next Run: {status.get('Next Run Time', 'Not scheduled')}")
        print(f"Last Result: {status.get('Last Result', 'N/A')}")
    else:
        print(f"Task '{manager.task_name}' is not configured.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='MT5 Data Collector - Task Scheduler Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create hourly task (every 1 hour)
  python scheduler.py --create hourly --interval 1

  # Create task every 30 minutes
  python scheduler.py --create minute --interval 30

  # Create daily tasks at specific times
  python scheduler.py --create daily --times 08:00,12:00,18:00

  # Delete all tasks
  python scheduler.py --delete

  # Check status
  python scheduler.py --status

  # Run task now
  python scheduler.py --run
'''
    )

    parser.add_argument(
        '--create',
        choices=['hourly', 'minute', 'daily'],
        help='Create scheduled task'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Interval for hourly/minute schedules'
    )
    parser.add_argument(
        '--times',
        type=str,
        default='06:00,12:00,18:00,23:00',
        help='Comma-separated times for daily schedule (HH:MM)'
    )
    parser.add_argument(
        '--days',
        type=str,
        default='MON,TUE,WED,THU,FRI',
        help='Days for daily schedule'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete scheduled task(s)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show task status'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run task immediately'
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default='MT5_DataCollector',
        help='Custom task name'
    )

    args = parser.parse_args()

    # Check if running on Windows
    if os.name != 'nt':
        print("Error: Task Scheduler is only available on Windows.")
        print("For Linux/macOS, use cron instead:")
        print("  crontab -e")
        print(f"  0 * * * * cd {PROJECT_DIR} && python mt5_collector.py")
        sys.exit(1)

    manager = TaskSchedulerManager(task_name=args.task_name)

    if args.status:
        print_status(manager)

    elif args.delete:
        count = manager.delete_all_tasks()
        print(f"Deleted {count} task(s)")

    elif args.run:
        if manager.run_task_now():
            print("Task triggered successfully")
        else:
            print("Failed to trigger task")

    elif args.create:
        if args.create == 'hourly':
            success = manager.create_hourly_task(interval_hours=args.interval)
        elif args.create == 'minute':
            success = manager.create_interval_task(interval_minutes=args.interval)
        elif args.create == 'daily':
            times = args.times.split(',')
            success = manager.create_daily_task(times=times, days=args.days)

        if success:
            print("\nTask created successfully!")
            print_status(manager)
        else:
            print("Failed to create task. Run as Administrator?")
            sys.exit(1)

    else:
        print_status(manager)
        print("\nUse --help for usage information")


if __name__ == "__main__":
    main()
